import argparse
import logging
import time
import dataclasses
import json

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
from src.model.mhnn_btsp_linear import *
from src.tools.utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for fine-tuning BTSP component of MHNN"""

    # static parameters
    config_file: str = None

    # BTSP parameters
    hash_ratio: float | list[float] = -1.0
    fw: float | list[float] = -1.0  # connection ratio
    wta_num: int | list[int] = -1  # number of winners(hash length)
    fq_constant: float | list[float] = -1.0  # fq = fq_constant * wta_num / output_dim
    fast_btsp: bool | list[bool] = False  # whether to use fast BTSP

    # sample parameters
    sample_length: int | list[int] = -1

    # experiment parameters
    train_exp_idx: str | list[str] = None
    test_exp_idx: str | list[str] = None
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

        self.results = self.mp_manager.dict()
        self.results["test_acc"] = self.mp_manager.list()
        self.results["test_recall"] = self.mp_manager.list()
        self.results["test_precision"] = self.mp_manager.list()
        self.results["btsp_fq"] = self.mp_manager.list()
        self.results["btsp_input_dim"] = self.mp_manager.list()
        self.results["experiment_index"] = self.mp_manager.list()

    def load_dataset(self):
        """Load dataset JIT, no need to load dataset here"""
        pass

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

        # unpack BTSP parameters
        btsp_hash_ratio = parameters.hash_ratio
        btsp_fw = parameters.fw
        btsp_wta_num = parameters.wta_num
        btsp_fq_constant = parameters.fq_constant
        fast_btsp = parameters.fast_btsp

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

        data_path = str(config["data_path"])
        snn_path = str(config["snn_path"])
        hnn_path = str(config["hnn_path"])
        # model_saving_file_name = str(config['model_saving_file_name'])

        w_fps = int(config["w_fps"])
        w_gps = int(config["w_gps"])
        w_dvs = int(config["w_dvs"])
        w_head = int(config["w_head"])
        w_time = int(config["w_time"])

        # device_id = str(config['device_id'])

        normalize = torchvision.transforms.Normalize(
            mean=[0.3537, 0.3537, 0.3537], std=[0.3466, 0.3466, 0.3466]
        )

        # approximately 24GB VRAM usage
        batch_size = int(np.floor(14 * 20 / (parameters.hash_ratio)))
        if not parameters.fast_btsp:
            batch_size = int(np.ceil(2 * 20 / (parameters.hash_ratio)))

        train_loader = Data(
            data_path,
            batch_size=batch_size,
            exp_idx=train_exp_idx,
            is_shuffle=True,
            normalize=normalize,
            nclass=num_class,
            seq_len_aps=seq_len_aps,
            seq_len_dvs=seq_len_dvs,
            seq_len_gps=seq_len_gps,
            seq_len_head=seq_len_head,
            seq_len_time=seq_len_time,
        )

        test_loader = Data(
            data_path,
            batch_size=batch_size,
            exp_idx=test_exp_idx,
            is_shuffle=True,
            normalize=normalize,
            nclass=num_class,
            seq_len_aps=seq_len_aps,
            seq_len_dvs=seq_len_dvs,
            seq_len_gps=seq_len_gps,
            seq_len_head=seq_len_head,
            seq_len_time=seq_len_time,
        )

        mhnn = MHNNBTSPLinear(
            device=device,
            cnn_arch=cnn_arch,
            num_epoch=num_epoch,
            batch_size=batch_size,
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
            btsp_wta_num=btsp_wta_num,
            btsp_fq_constant=btsp_fq_constant,
            btsp_hash_ratio=btsp_hash_ratio,
            btsp_fw=btsp_fw,
            fast_btsp=fast_btsp,
            seq_len_sample=parameters.sample_length,
        )

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
        print(device)

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
        
        # # first train btsp separately(one-shot training)
        # mhnn.freeze_lc()
        # for inputs, target in train_loader:
        #     mhnn.train()
        #     outputs, outs = mhnn(inputs, epoch=0)
        # mhnn.unfreeze_lc()

        # # train lc and test whole network
        # mhnn.freeze_btsp()
        
        start_time = time.time()
        for epoch in range(num_epoch):

            ###############
            ## for training
            ###############

            running_loss = 0.0
            counts = 1.0
            acc1_record, acc5_record, acc10_record = 0.0, 0.0, 0.0
            while iters < num_iter:
                mhnn.train()
                optimizer.zero_grad()

                try:
                    inputs, target = next(train_iters)
                except StopIteration:
                    print("StopIteration")
                    train_iters = iter(train_loader)
                    inputs, target = next(train_iters)

                outputs, outs = mhnn(inputs, epoch=epoch)

                class_loss = criterion(outputs, target.to(device))

                ## no sparse loss for BTSP
                # sparse_loss = 0.0
                # w_module = (w_fps, w_dvs, w_gps, w_gps)
                # for i, l in enumerate(outs):
                #     sparse_loss += (l.mean() - r) ** 2 * w_module[i]

                # loss = class_loss + sparse_lambdas * sparse_loss
                loss = class_loss

                loss.backward()

                optimizer.step()

                running_loss += loss.cpu().item()

                acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
                acc1, acc5, acc10 = (
                    acc1 / len(outputs),
                    acc5 / len(outputs),
                    acc10 / len(outputs),
                )

                acc1_record += acc1
                acc5_record += acc5
                acc10_record += acc10

                counts += 1
                iters += 1.0
            iters = 0

            record["loss"].append(loss.item())
            print("\n\nTime elaspe:", time.time() - start_time, "s")
            lr_schedule.step()
            print(
                "Training epoch %.1d, training loss :%.4f, training Top1 acc: %.4f, training Top5 acc: %.4f"
                % (
                    epoch,
                    running_loss / (num_iter),
                    acc1_record / counts,
                    acc5_record / counts,
                )
            )

            ##############
            ## for testing
            ##############

            running_loss = 0.0
            mhnn.eval()

            with torch.no_grad():

                acc1_record, acc5_record, acc10_record = 0.0, 0.0, 0.0
                counts = 1.0

                # progress_bar_testing.reset()

                for batch_idx, (inputs, target) in enumerate(test_loader):

                    outputs, _ = mhnn(inputs, epoch=epoch)

                    loss = criterion(outputs.cpu(), target)

                    running_loss += loss.item()
                    acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
                    acc1, acc5, acc10 = (
                        acc1 / len(outputs),
                        acc5 / len(outputs),
                        acc10 / len(outputs),
                    )
                    acc1_record += acc1
                    acc5_record += acc5
                    acc10_record += acc10

                    counts += 1
                    outputs = outputs.cpu()

                    test_acc(outputs.argmax(1), target)
                    test_recall(outputs.argmax(1), target)
                    test_precision(outputs.argmax(1), target)

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


parser = argparse.ArgumentParser(description="mhnn", argument_default=argparse.SUPPRESS)
parser.add_argument(
    "--config_file", type=str, required=True, help="Configure file path"
)


if __name__ == "__main__":
    options = parser.parse_args()

    meta_params = MetaParams(
        config_file=options.config_file,
        sample_length=3,
        hash_ratio=[15,20],
        fw=[0.005],
        wta_num=[100,200],
        fq_constant=[0.01,0.03],
        fast_btsp=True,
        experiment_index=0,
        experiment_name="mhnn_btsp_linear_debug",
    )

    experiment = cuda_distributed_experiment.CudaDistributedExperiment(
        ExpMHNNBTSPFinetune(meta_params), "max", backend="torch"
    )
    experiment.run()
    experiment.evaluate()
