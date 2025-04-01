import argparse
import logging
import time
import dataclasses
import json

import shutil

import tqdm

import torch
import torch.utils.model_zoo
import torchmetrics
import torch.multiprocessing as mp

import pandas as pd

from auto_experiment import cuda_distributed_experiment, auto_experiment
from auto_utils import parameterization
from auto_utils import logging as auto_logging

from src.config.config_utils import *
from src.model.mhnn_btsp import *
from src.tools.utils import *

from torch.profiler import profile, record_function, ProfilerActivity

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

@dataclasses.dataclass
class NormalizationConfig:
    
    visual_signal_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | list[tuple[tuple[float, float, float], tuple[float, float, float]]] = ((0.3537, 0.3537, 0.3537), (0.3466, 0.3466, 0.3466)) # default value set by NeuroGPR project
    
    w_cnn_pixel_normalization: bool | list[bool] = False
    
    cnn_pixel_mean_tensor_path: str = None

@dataclasses.dataclass
class TemporalInputConfig:
    
    spike_train_sample_length: int | list[int] = 3

    # avaliable options: flatten, as_pattern, average
    temporal_dimension_handling: str | list[str] = "as_pattern"

@dataclasses.dataclass
class PostPreprocessingConfig:
    
    # refer to results/btsp_input_analysis_3_distribution/
    # format: (aps, dvs, gps, head, time)
    modality_scale_factor: tuple[float, float, float, float, float] | list[tuple[float, float, float, float, float]] = (1.0, 1.0, 1.0, 1.0, 1.0)

@dataclasses.dataclass
class MetaParams:
    """Meta parameters for fine-tuning BTSP component of MHNN"""

    # static parameters
    config_file: str = None
    
    # preprocessing parameters
    normalization_config: NormalizationConfig = None
    enable_modalities: list[list[str]] = None
    enable_cnn_wta: bool | list[bool] = False
    cnn_wta_ratio: float | list[float] = -1.0
    
    post_preprocessing_config: PostPreprocessingConfig = None

    # BTSP parameters
    binary_btsp: bool | list[bool] = True
    temporal_input_config: TemporalInputConfig = None
    hash_ratio: float | list[float] = -1.0
    fw: float | list[float] = -1.0  # connection ratio
    wta_ratio: float | list[float] = -1.0  # wta_ratio = wta_num / output_dim
    fq_constant: float | list[float] = -1.0  # fq = fq_constant * wta_num / output_dim
    fast_btsp: bool | list[bool] = False  # whether to use fast BTSP

    # experiment parameters
    train_exp_idx: str | list[str] = None
    test_exp_idx: str | list[str] = None
    partial_train: bool | list[bool] = False
    training_proportion: float | list[float] = 1.0 # randomly select a proportion of training data from the training set
    partial_test: bool | list[bool] = False # use a subset of the same proportion of the training data for testing
    train_batch_size: int = None
    test_batch_size: int = None
    worker_per_data_loader: int = 2
    redis_config: RedisConfig = None
    preload_list: tuple[int] = None
    experiment_index: int = -1
    experiment_name: str = "debug"


class ExpMHNNBTSPFinetune(auto_experiment.ExperimentInterface):

    mp_manager = mp.Manager()

    def __init__(self, meta_params: MetaParams):
        self.meta_params = meta_params

        self.experiment_folder = auto_logging.init_experiment_folder(
            "./results", self.meta_params.experiment_name, False
        )
        json.dump(
            dataclasses.asdict(self.meta_params),
            open(os.path.join(self.experiment_folder, "meta_params.json"), "w"),
            indent=4,
        )

        # copy the config file to the experiment folder
        config_file_path = os.path.join("./src", self.meta_params.config_file)
        shutil.copy(
            config_file_path, os.path.join(self.experiment_folder, "config.ini")
        )

        self.results = self.mp_manager.dict()
        self.results["test_acc"] = self.mp_manager.list()
        self.results["test_recall"] = self.mp_manager.list()
        self.results["test_precision"] = self.mp_manager.list()
        self.results["btsp_fq"] = self.mp_manager.list()
        self.results["btsp_input_dim"] = self.mp_manager.list()
        self.results["experiment_index"] = self.mp_manager.list()

    def load_dataset(self):
        """Preload the dataset if there is a Redis configuration in meta_params"""
        if self.meta_params.redis_config is not None:
            data_dir = str(get_config(self.meta_params.config_file, logger=log)["main"]["data_path"])
            dataset = preload_dataset_to_redis(data_dir=data_dir, exp_index_list=list(self.meta_params.preload_list), redis_config=self.meta_params.redis_config, reuse=True)

    def load_parameters(self):
        param_list = parameterization.recursive_iterate_dataclass(self.meta_params)
        param_list = parameterization.set_experiment_index(
            param_list, self.meta_params.experiment_index
        )
        parameterization.save_dataclasses_to_csv(param_list, self.experiment_folder)
        return param_list

    def execute_experiment_process(self, parameters: MetaParams, dataset):

        # check torch visible devices
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            # for debugging
            device = "cuda:" + str(torch.cuda.current_device())

        # unpack experiment parameters
        btsp_hash_ratio = parameters.hash_ratio
        btsp_fw = parameters.fw
        btsp_wta_ratio = parameters.wta_ratio
        btsp_fq_constant = parameters.fq_constant
        fast_btsp = parameters.fast_btsp
        binary_btsp = parameters.binary_btsp
        enable_cnn_wta = parameters.enable_cnn_wta
        cnn_wta_ratio = parameters.cnn_wta_ratio
        
        spike_train_sample_len = parameters.temporal_input_config.spike_train_sample_length
        temporal_dim_handling = parameters.temporal_input_config.temporal_dimension_handling
        
        w_aps_norm = parameters.normalization_config.w_cnn_pixel_normalization
        cnn_pixel_mean_tensor_path = parameters.normalization_config.cnn_pixel_mean_tensor_path
        
        post_preprocessing_config = parameters.post_preprocessing_config

        # load static parameters
        config_file = parameters.config_file
        config = get_config(config_file, logger=log)["main"]

        config = get_config(options.config_file, logger=log)["main"]
        config = update_config(config, options.__dict__)

        cnn_arch = str(config["cnn_arch"])
        num_epoch = int(config["num_epoch"])
        # batch_size = int(config['batch_size'])
        num_class = int(config["num_class"])
        rnn_num = int(config["rnn_num"])
        cann_num = int(config["cann_num"])
        reservoir_num = int(config["reservoir_num"])
        num_iter = int(config["num_iter"])
        spiking_threshold = float(config["spiking_threshold"])
        sparse_lambdas = int(config["sparse_lambdas"])
        lr = float(config["lr"])
        r = float(config["r"])

        ann_pre_load = get_bool_from_config(config, "ann_pre_load")
        snn_pre_load = get_bool_from_config(config, "snn_pre_load")
        re_trained = get_bool_from_config(config, "re_trained")

        seq_len_aps = int(config["seq_len_aps"])
        seq_len_gps = int(config["seq_len_gps"])
        seq_len_dvs = int(config["seq_len_dvs"])
        seq_len_head = int(config["seq_len_head"])
        seq_len_time = int(config["seq_len_time"])

        dvs_expand = int(config["dvs_expand"])
        expand_len = least_common_multiple(
            [seq_len_aps, seq_len_dvs * dvs_expand, seq_len_gps]
        )

        test_exp_idx = []
        if parameters.test_exp_idx is None:
            input_test_exp_idx = config["test_exp_idx"]
        else:
            input_test_exp_idx = parameters.test_exp_idx
        for idx in input_test_exp_idx:
            if idx != ",":
                test_exp_idx.append(int(idx))

        train_exp_idx = []
        if parameters.train_exp_idx is None:
            input_train_exp_idx = config["train_exp_idx"]
        else:
            input_train_exp_idx = parameters.train_exp_idx
        for idxt in input_train_exp_idx:
            if idxt != ",":
                train_exp_idx.append(int(idxt))
                
        if (train_exp_idx != test_exp_idx) and parameters.partial_test:
            raise ValueError("Partial test is only available when train_exp_idx == test_exp_idx")

        data_path = str(config["data_path"])
        snn_path = str(config["snn_path"])
        hnn_path = str(config["hnn_path"])
        # model_saving_file_name = str(config['model_saving_file_name'])

        w_fps = int(config["w_fps"])
        w_gps = int(config["w_gps"])
        w_dvs = int(config["w_dvs"])
        w_head = int(config["w_head"])
        w_time = int(config["w_time"])
        
        if "fps" in parameters.enable_modalities:
            w_fps = 1
        else:
            w_fps = 0
        if "gps" in parameters.enable_modalities:
            w_gps = 1
        else:
            w_gps = 0
        if "dvs" in parameters.enable_modalities:
            w_dvs = 1
        else:
            w_dvs = 0
        if "head" in parameters.enable_modalities:
            w_head = 1
        else:
            w_head = 0
        if "time" in parameters.enable_modalities:
            w_time = 1
        else:
            w_time = 0

        # device_id = str(config['device_id'])

        normalize = torchvision.transforms.Normalize(
            mean=list(parameters.normalization_config.visual_signal_normalization[0]), std=list(parameters.normalization_config.visual_signal_normalization[1])
        )

        if parameters.train_batch_size is not None:
            train_batch_size = parameters.train_batch_size
        else:
            train_batch_size = int(config["batch_size"])
        
        if parameters.test_batch_size is not None:
            test_batch_size = parameters.test_batch_size
        else:
            test_batch_size = int(config["batch_size"])

        train_loader = Data(
            data_path,
            batch_size=train_batch_size,
            exp_idx=train_exp_idx,
            is_shuffle=True,
            normalize=normalize,
            nclass=num_class,
            seq_len_aps=seq_len_aps,
            seq_len_dvs=seq_len_dvs,
            seq_len_gps=seq_len_gps,
            seq_len_head=seq_len_head,
            seq_len_time=seq_len_time,
            worker_per_data_loader=parameters.worker_per_data_loader,
            redis_config=parameters.redis_config,
            w_aps_norm=w_aps_norm,
            aps_norm_tensor_path=cnn_pixel_mean_tensor_path,
        )
        
        if parameters.partial_test:
            test_loader = train_loader
        else:
            test_loader = Data(
                data_path,
                batch_size=test_batch_size,
                exp_idx=test_exp_idx,
                is_shuffle=True,
                normalize=normalize,
                nclass=num_class,
                seq_len_aps=seq_len_aps,
                seq_len_dvs=seq_len_dvs,
                seq_len_gps=seq_len_gps,
                seq_len_head=seq_len_head,
                seq_len_time=seq_len_time,
                worker_per_data_loader=parameters.worker_per_data_loader,
                redis_config=parameters.redis_config,
                w_aps_norm=w_aps_norm,
                aps_norm_tensor_path=cnn_pixel_mean_tensor_path
            )

        mhnn = MHNNBTSP(
            device=device,
            cnn_arch=cnn_arch,
            num_epoch=num_epoch,
            num_class=num_class,
            rnn_num=rnn_num,
            cann_num=cann_num,
            reservoir_num=reservoir_num,
            spiking_threshold=spiking_threshold,
            sparse_lambdas=sparse_lambdas,
            r=r,
            lr=lr,
            w_fps=w_fps,
            w_gps=w_gps,
            w_dvs=w_dvs,
            w_head=w_head,
            w_time=w_time,
            seq_len_aps=seq_len_aps,
            seq_len_gps=seq_len_gps,
            seq_len_dvs=seq_len_dvs,
            seq_len_head=seq_len_head,
            seq_len_time=seq_len_time,
            dvs_expand=dvs_expand,
            expand_len=expand_len,
            train_exp_idx=train_exp_idx,
            test_exp_idx=test_exp_idx,
            data_path=data_path,
            snn_path=snn_path,
            hnn_path=hnn_path,
            num_iter=num_iter,
            ann_pre_load=ann_pre_load,
            snn_pre_load=snn_pre_load,
            re_trained=re_trained,
            btsp_wta_ratio=btsp_wta_ratio,
            btsp_fq_constant=btsp_fq_constant,
            btsp_hash_ratio=btsp_hash_ratio,
            btsp_fw=btsp_fw,
            fast_btsp=fast_btsp,
            temporal_dim_handling=temporal_dim_handling,
            spike_train_sample_len=spike_train_sample_len,
            enable_cnn_wta=enable_cnn_wta,
            cnn_wta_ratio=cnn_wta_ratio,
            binary_btsp=binary_btsp,
            post_preprocessing_norm=post_preprocessing_config,
        )

        print("Initializing CANN")
        mhnn.cann_init(
            np.concatenate(
                (
                    train_loader.dataset.data_pos[0],
                    train_loader.dataset.data_head[0][:, 1].reshape(-1, 1),
                ),
                axis=1,
            )
        )

        mhnn.to(device)

        print("Start training")
        optimizer = torch.optim.Adam(mhnn.parameters(), lr)

        lr_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )

        criterion = nn.CrossEntropyLoss()

        record = {}
        record["loss"], record["top1"], record["top5"], record["top10"] = [], [], [], []
        best_test_acc1, best_test_acc5, best_recall, best_test_acc10 = 0.0, 0.0, 0, 0

        train_iters = iter(train_loader)
        iters = 0
        start_time = time.time()
        # print(device)

        best_recall = 0.0
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)

        test_recall = torchmetrics.Recall(
            task="multiclass", average="none", num_classes=num_class
        )
        test_precision = torchmetrics.Precision(
            task="multiclass", average="none", num_classes=num_class
        )

        # progress_bar_training = tqdm.tqdm(range(len(train_loader)), desc='Training')
        # progress_bar_testing = tqdm.tqdm(range(len(test_loader)), desc='Testing')

        # keep for now, num_epoch should be 1
        if num_epoch != 1:
            print("Overwriting num_epoch to 1 for one-shot learning")
            num_epoch = 1
        for epoch in range(num_epoch):

            ###############
            ## for training
            ###############

            # BTSP is a one-shot online learning algorithm
            # for each training data, a single forward pass is enough
            # no need for backpropagation

            # also because of this, the re-train option of cnn and snn is no longer available

            mhnn.train()
            hash_table = (
                None  # BTSP hashing result, (sample index, seq_length, channel_num)
            )
            hash_target = None  # the target(category) of the hashing result, a 1-d tensor that maps sample index to category

            with torch.no_grad():

                # progress_bar_training.reset()
                
                # get batch number
                batch_num = len(train_loader)
                
                training_batches = np.arange(batch_num)
                np.random.shuffle(training_batches)
                
                drop_num = int(batch_num * (1 - parameters.training_proportion))
                training_batches = training_batches[drop_num:]
                
                hash_table = None
                hash_target = None

                for batch_idx, (inputs, target) in enumerate(train_loader):
                    # if parameters.partial_train:
                    #     if batch_idx not in training_batches:
                    #         continue

                    outputs, _ = mhnn(inputs, epoch=epoch)

                    # for single forward pass, no need for calculating loss

                    # append the hashing result--a matrix shaped (seq_length, channel_num) to the hash_table
                    # append the target to the hash_target
                    if hash_table is None:
                        hash_table = outputs
                        hash_target = target
                    else:
                        hash_table = torch.cat((hash_table, outputs), dim=0)
                        hash_target = torch.cat((hash_target, target), dim=0)

                    # progress_bar_training.update(1)

            ##############
            ## for testing
            ##############

            running_loss = 0.0
            mhnn.eval()

            with torch.no_grad():

                acc1_record, acc5_record, acc10_record = 0.0, 0.0, 0.0
                counts = 1.0

                # progress_bar_testing.reset()
                testing_batches = training_batches

                for batch_idx, (inputs, target) in enumerate(test_loader):
                    
                    if parameters.partial_test:
                        if batch_idx not in testing_batches:
                            continue

                    outputs, _ = mhnn(inputs, epoch=epoch)

                    # loss = criterion(outputs.cpu(), target)

                    # find the nearest hash labels
                    labels = find_nearest_hash(
                        outputs, hash_table, hash_target, device=device
                    )

                    # running_loss += loss.item()
                    # acc1, acc5, acc10 = accuracy(labels.cpu(), target, topk=(1, 5, 10))
                    # acc1, acc5, acc10 = acc1 / len(labels), acc5 / len(labels), acc10 / len(labels)
                    # acc1_record += acc1
                    # acc5_record += acc5
                    # acc10_record += acc10

                    counts += 1
                    labels = labels.cpu()

                    assert labels.shape[0] == target.shape[0]

                    test_acc(labels, target)
                    test_recall(labels, target)
                    test_precision(labels, target)

                    compute_num = 100
                    if batch_idx % compute_num == 0 and batch_idx > 0:
                        # print('time:', (time.time() - start_time) / compute_num)
                        start_time = time.time()

                    # progress_bar_testing.update(1)

            total_acc = test_acc.compute().mean().item()
            total_recall = test_recall.compute().mean().item()
            total_precison = test_precision.compute().mean().item()

            # report the accuracy, recall, and precision
            # print('Test Acc: ', total_acc, 'Test Recall: ', total_recall, 'Test Precision: ', total_precison)

            # append the accuracy, recall, and precision to the results
            self.results["test_acc"].append(total_acc)
            self.results["test_recall"].append(total_recall)
            self.results["test_precision"].append(total_precison)
            self.results["btsp_fq"].append(mhnn.btsp_fq)
            self.results["btsp_input_dim"].append(mhnn.btsp_input_dim)
            self.results["experiment_index"].append(parameters.experiment_index)

            test_precision.reset()
            test_recall.reset()
            test_acc.reset()

    def summarize_results(self):
        # convert the result_dict to dict
        self.results = dict(self.results)
        for key in self.results.keys():
            self.results[key] = list(self.results[key])

        results = pd.DataFrame(self.results)
        # save the results to csv
        results.to_csv(os.path.join(self.experiment_folder, "results.csv"), index=False)
        # merge with params and save
        meta_params_df = pd.read_csv(os.path.join(self.experiment_folder, "params.csv"))
        results = pd.merge(meta_params_df, results, on=["experiment_index"])
        results.to_csv(
            os.path.join(self.experiment_folder, "params_and_results.csv"), index=True
        )
        
        # if there is a redis configuration, empty the redis database
        r = redis.Redis(
            host=self.meta_params.redis_config.redis_host,
            port=self.meta_params.redis_config.redis_port,
            db=self.meta_params.redis_config.redis_db_num,
        )
        r.flushdb()


parser = argparse.ArgumentParser(description="mhnn", argument_default=argparse.SUPPRESS)
parser.add_argument(
    "--config_file", type=str, required=True, help="Configure file path"
)


if __name__ == "__main__":
    options = parser.parse_args()
    
    redis_config = RedisConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db_num=0,
    )
    
    normalization_config = NormalizationConfig(
        w_cnn_pixel_normalization= [True, False],
        cnn_pixel_mean_tensor_path= "/root/autodl-tmp/rht/BTSP_task4_neurogpr/data/norm_tensor_forest"
    )
    
    temporal_input_config = TemporalInputConfig(
        spike_train_sample_length=3,
        temporal_dimension_handling=["flatten", "average"],
    )
    
    post_preprocessing_config = PostPreprocessingConfig(
        modality_scale_factor=[(1.0, 1.0, 1.0, 1.0, 1.0), (0.66, 25.0, 25.0, 25.0, 25.0)]
    )

    meta_params = MetaParams(
        config_file=options.config_file,
        normalization_config=normalization_config,
        enable_modalities=[
        [
            "fps",
            "gps",
            "head",
            "time"
        ],
        ],
        temporal_input_config=temporal_input_config,
        enable_cnn_wta=[True],
        cnn_wta_ratio=[0.1, 0.25, 0.5, 0.75, 1.0],
        # cnn_wta_ratio=[0.1],
        post_preprocessing_config=post_preprocessing_config,
        binary_btsp=[False],
        hash_ratio=[
        1.5
        ], # static, the ratio of CA3 neurons to CA1 neurons in human brain
        fw=[0.6,1.0], # static, the connection ratio between CA3 and CA1 neurons
        wta_ratio=[0.001,0.01,0.1],
        # wta_ratio=[0.01],
        fq_constant=[
        0.01,0.1,1.0],
        # fq_constant=[0.05],
        fast_btsp=True,
        train_exp_idx=["1"],
        test_exp_idx=["2"],
        train_batch_size=256,
        test_batch_size=256,
        worker_per_data_loader=2,
        training_proportion=[1.0],
        partial_test=[False],
        redis_config=redis_config,
        preload_list=(1,2),
        experiment_index=0,
        experiment_name="mission5_btsp_test3_forest_setting_ft",
    )
    
    #experiment = auto_experiment.SimpleBatchExperiment(ExpMHNNBTSPFinetune(meta_params),1)

    experiment = cuda_distributed_experiment.CudaDistributedExperiment(
        ExpMHNNBTSPFinetune(meta_params), "max", backend="torch"
    )
    experiment.run()
    experiment.evaluate()
