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

if __name__ == "__main__":

    normalize = torchvision.transforms.Normalize(
        mean=[0.3537, 0.3537, 0.3537], std=[0.3466, 0.3466, 0.3466]
    )

    train_loader = Data(
        "/root/autodl-tmp/data/floor3_v",
        batch_size=1,
        exp_idx=[1],
        is_shuffle=True,
        normalize=normalize,
        nclass=100,
        seq_len_aps=3,
        seq_len_dvs=3,
        seq_len_gps=3,
        seq_len_head=3,
        seq_len_time=3,
    )
    
    # get the first batch of the dataset
    for i, (data, target) in enumerate(train_loader):
        
        # get vision modality
        aps = data[0]
        
        # print the aps tensor
        print(aps)
        
        if i == 0:
            break