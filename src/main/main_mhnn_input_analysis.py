import argparse
import dataclasses
import json
import logging
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.utils.model_zoo
import torchmetrics
import tqdm
from auto_experiment import auto_experiment, cuda_distributed_experiment
from auto_utils import logging as auto_logging
from auto_utils import parameterization

from src.config.config_utils import *
from src.model.mhnn_preprocessor_only import *
from src.tools.utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for fine-tuning BTSP component of MHNN"""

    # static parameters
    config_file: str = None

    # BTSP parameters
    flatten_temporal_input: bool | list[bool] = False

    # experiment parameters
    target_dataset: str | list[str] = None
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

        train_exp_idx = []
        if parameters.target_dataset is None:
            input_train_exp_idx = config["train_exp_idx"]
        else:
            input_train_exp_idx = parameters.target_dataset
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
        batch_size = int(np.floor(7 * 20 / (1)))
        if parameters.flatten_temporal_input:
            batch_size = int(np.ceil(batch_size / seq_len_aps)) # TODO: isolate sample length interface

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

        mhnn = MHNNPreprocessorOnly(
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
            data_path=data_path,
            snn_path=snn_path,
            hnn_path=hnn_path,
            num_iter=num_iter,
            ann_pre_load=ann_pre_load,
            snn_pre_load=snn_pre_load,
            re_trained=re_trained,
            flatten_temporal_input=parameters.flatten_temporal_input,
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

        # preprocessed dataset analysis
        with torch.no_grad():
            
            # fetch all preprocessed inputs
            vision_all = None
            for batch_idx, (inputs, target) in enumerate(train_loader):
                
                # out1-5, combined_input: [seq_len, batch_size, channel]
                # out1-5: vision(cnn), speed(snn), position(cann), time(cann), direction(cann)
                # combined_input: unified sampled input
                out1, out2, out3, out4, out5, combined_input = mhnn(inputs)
                
                vision_temp = out1
                speed_temp = out2
                position_temp = out3
                time_temp = out4
                direction_temp = out5
                sampled_input_temp = combined_input
                
                # concatenate
                if vision_all is None:
                    vision_all = vision_temp
                    speed_all = speed_temp
                    position_all = position_temp
                    time_all = time_temp
                    direction_all = direction_temp
                    sampled_input_all = sampled_input_temp
                else:
                    vision_all = torch.cat((vision_all, vision_temp), 1)
                    speed_all = torch.cat((speed_all, speed_temp), 1)
                    position_all = torch.cat((position_all, position_temp), 1)
                    time_all = torch.cat((time_all, time_temp), 1)
                    direction_all = torch.cat((direction_all, direction_temp), 1)
                    sampled_input_all = torch.cat((sampled_input_all, sampled_input_temp), 1)
            
            # print(vision_all.size())
            # print('batch size:',batch_size)
            # print('batch idx:',batch_idx)
            # NOTE: drop_last is set in the dataloader of NeuroGPR
            # vision_all shape: [seq_len, batch_size * (batch_idx + 1), channel]
            total_len = batch_size * (batch_idx + 1)
            assert vision_all.size(1) == total_len
            
            # all snn input should be 0
            print(torch.sum(speed_all))
            assert torch.sum(speed_all) == 0
            
            # input without snn
            input_without_snn = torch.cat((vision_all, position_all, time_all, direction_all), 2)
            
            if parameters.flatten_temporal_input:
                # reshape to [batch, seq_len * channel]
                input_without_snn = input_without_snn.permute(1, 0, 2).reshape(total_len, -1)
                assert input_without_snn.size(0) == total_len
            else:
                # reshape to [seq_len * batch, channel]
                input_without_snn = input_without_snn.permute(1, 0, 2).reshape(-1, input_without_snn.size(2))
                assert input_without_snn.size(0) == total_len * seq_len_aps
            
            # calculate the sparsity of each input
            input_without_snn_sparsity = torch.sum(input_without_snn, dim=-1) / input_without_snn.size(-1)
            
            if parameters.flatten_temporal_input:
                flatten_flag = 'flatten'
            else:
                flatten_flag = 'not_flatten'

            # plot the sparsity distribution
            sns.set_style("darkgrid")
            sns.displot(input_without_snn_sparsity.cpu().numpy(), binwidth=0.02)
            plt.xlim(0, 1)
            plt.xlabel('Sparsity')
            plt.ylabel('Count')
            # save as svg
            plt.savefig(os.path.join(self.experiment_folder, f"sparsity_distribution_{train_exp_idx[0]}_{flatten_flag}.svg"))
            # close the plot
            plt.close()
            
            # save the sparsity distribution
            # sparsity_df = pd.DataFrame(input_without_snn_sparsity.cpu().numpy())
            # sparsity_df.to_csv(os.path.join(self.experiment_folder, f"sparsity_distribution_{train_exp_idx[0]}_{flatten_flag}.csv"), index=False)
            
            # calculate the dice coefficient between each pair of inputs
            dc_matrix = dice_coefficient_matrix(input_without_snn)
            
            # flatten the dc matrix and remove the -1 elements
            dc_matrix = dc_matrix.flatten()
            dc_matrix = dc_matrix[dc_matrix != -1]
            
            # plot the dice coefficient distribution
            sns.displot(dc_matrix.cpu().numpy(), binwidth=0.02)
            plt.xlim(0, 1)
            plt.xlabel('Dice Coefficient')
            plt.ylabel('Count')
            # save as svg
            plt.savefig(os.path.join(self.experiment_folder, f"dice_coefficient_distribution_{train_exp_idx[0]}_{flatten_flag}.svg"))
            # close the plot
            plt.close()
            
            # save the dice coefficient distribution
            # dc_df = pd.DataFrame(dc_matrix.cpu().numpy())
            # dc_df.to_csv(os.path.join(self.experiment_folder, f"dice_coefficient_distribution_{train_exp_idx[0]}_{flatten_flag}.csv"), index=False)
            
            return


    def summarize_results(self):
        pass
        # convert the result_dict to dict
        # self.results = dict(self.results)
        # for key in self.results.keys():
        #     self.results[key] = list(self.results[key])

        # results = pd.DataFrame(self.results)
        # # save the results to csv
        # results.to_csv(os.path.join(self.experiment_folder, "results.csv"), index=False)
        # # merge with params and save
        # meta_params_df = pd.read_csv(os.path.join(self.experiment_folder, "params.csv"))
        # results = pd.merge(meta_params_df, results, on=["experiment_index"])
        # results.to_csv(
        #     os.path.join(self.experiment_folder, "params_and_results.csv"), index=True
        # )


parser = argparse.ArgumentParser(description="mhnn", argument_default=argparse.SUPPRESS)
parser.add_argument(
    "--config_file", type=str, required=True, help="Configure file path"
)


if __name__ == "__main__":
    options = parser.parse_args()

    meta_params = MetaParams(
        config_file=options.config_file,
        flatten_temporal_input=[True, False],
        target_dataset=['1','3','5','7','9'],
        experiment_index=0,
        experiment_name="btsp_input_analysis",
    )

    experiment = cuda_distributed_experiment.CudaDistributedExperiment(
        ExpMHNNBTSPFinetune(meta_params), "max", backend="torch"
    )
    experiment.run()
    experiment.evaluate()
    print("Experiment finished.")
