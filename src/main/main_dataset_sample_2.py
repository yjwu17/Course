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

class FindMeanTensor(auto_experiment.ExperimentInterface):
    
    def __init__(self, meta_params: MetaParams):
        self.meta_params = meta_params

        exp_name = "mission5_forest_dataset_mean_tensor"
        self.save_path = auto_logging.init_experiment_folder("results",  exp_name, False)
    
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
            "/root/autodl-tmp/data/forest_v",
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
        
        # iterate over the dataset and calculate the running mean
        mean_vector = None
        count = 0
        for i, (data, target) in enumerate(iterable=train_loader):
            data = data[0].to(device)
            batch_size = data.size(0)
            if mean_vector is None:
                mean_vector = data.sum(dim=0) / batch_size
            else:
                mean_vector = (mean_vector * count + data.sum(dim=0)) / (count + batch_size)
            count += batch_size
        # save the mean vector, name by chunk index
        torch.save(mean_vector, os.path.join(self.save_path, f"{chunk}_3.pt"))
    
    def summarize_results(self):
        pass

if __name__ == "__main__":
    
    test_params = MetaParams(
        [1,2]
    )
    
    experiment = cuda_distributed_experiment.CudaDistributedExperiment(FindMeanTensor(test_params), "max")
    experiment.run()
    experiment.evaluate()
    