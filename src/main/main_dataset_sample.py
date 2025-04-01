import argparse
import logging
import time
import dataclasses
import json
import gc

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

@dataclasses.dataclass
class MetaParams:
    chunk_idx: int | list[int] = -1

class FindMeanStd(auto_experiment.ExperimentInterface):
    
    def __init__(self, meta_params: MetaParams):
        self.meta_params = meta_params

        exp_name = "Corridor_dataset_RGB_mean_std"
        self.save_path = auto_logging.init_experiment_folder("results",  exp_name, False)

        # create results.csv file
        self.mp_manager = mp.Manager()
        self.results = self.mp_manager.dict()
        self.results["chunk_idx"] = self.mp_manager.list()
        self.results["data_number"] = self.mp_manager.list()
        self.results["r_mean"] = self.mp_manager.list()
        self.results["r_std"] = self.mp_manager.list()
        self.results["g_mean"] = self.mp_manager.list()
        self.results["g_std"] = self.mp_manager.list()
        self.results["b_mean"] = self.mp_manager.list()
        self.results["b_std"] = self.mp_manager.list()
    
    def load_parameters(self):
        param_list = parameterization.recursive_iterate_dataclass(self.meta_params)
        return param_list
    
    def load_dataset(self):
        pass
    
    def execute_experiment_process(self, parameters: MetaParams, dataset):
        # check torch visible devices
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            # for debugging
            device = "cuda:" + str(torch.cuda.current_device())
        
        chunk = parameters.chunk_idx

        # disable normalization
        normalize = torchvision.transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
        )
        train_loader = Data(
            "/root/autodl-tmp/data/floor3_v",
            batch_size=256,
            exp_idx=[chunk],
            is_shuffle=True,
            normalize=normalize,
            nclass=100,
            seq_len_aps=3,
            seq_len_dvs=3,
            seq_len_gps=3,
            seq_len_head=3,
            seq_len_time=3,
        )
        # initialize lists to store r, g, b values
        mean_results = []
        std_results = []
        # get data number
        data_num = len(train_loader.dataset)
        # calculate r,g,b mean and std separately to reduce memory usage
        for rgb_index in range(3):
            channel_values = []
            for i, (data, target) in enumerate(train_loader):
                # get vision modality
                aps = data[0].to(device)
                channel = aps[:, :, rgb_index, :, :]
                # append values to lists
                channel_values.append(channel)
            # concatenate all values
            channel_values = torch.cat(channel_values, 0)
            # calculate mean and std
            mean = channel_values.mean().item()
            std = channel_values.std().item()
            # append results to lists
            mean_results.append(mean)
            std_results.append(std)
        
        # append results
        self.results["chunk_idx"].append(chunk)
        self.results["data_number"].append(data_num)
        self.results["r_mean"].append(mean_results[0])
        self.results["r_std"].append(std_results[0])
        self.results["g_mean"].append(mean_results[1])
        self.results["g_std"].append(std_results[1])
        self.results["b_mean"].append(mean_results[2])
        self.results["b_std"].append(std_results[2])
        
    
    def summarize_results(self):
        # convert the result_dict to dict
        self.results = dict(self.results)
        for key in self.results.keys():
            self.results[key] = list(self.results[key])

        results = pd.DataFrame(self.results)

        # save results to csv
        results.to_csv(f"{self.save_path}/results.csv", index=False)
        print("Results saved to csv file.")

if __name__ == "__main__":
    
    test_params = MetaParams(
        [1,3,5,7,9]
    )
    
    experiment = cuda_distributed_experiment.CudaDistributedExperiment(FindMeanStd(test_params), "max")
    experiment.run()
    experiment.evaluate()
    