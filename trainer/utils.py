#!/usr/bin/python3

"""Utils module

Implements helper functions.

"""

# Import libraries
import random

import numpy as np
import torch

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig


class AverageMeter:
    """Class for Computing and storing the average and current value
    """
    
    def __init__(self):
        """Init method of class
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset method
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        """Update method

        Args:
            val (int): input of value
            count (int, optional): number of values. Defaults to 1.
        """        
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count




def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """Patches configs if cuda is not available

    Args:
        epoch_num_to_set (int, optional): Number of times the whole dataset will be passed through the network. Defaults to TrainerConfig.epoch_num.
        batch_size_to_set (int, optional): Amount of data to pass through the network at each forward-backward iteration. Defaults to DataloaderConfig.batch_size.

    Returns:
        dataloader_config, trainer_config: returns patched dataloader_config and trainer_config
    """    
    
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set)
    return dataloader_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    """Setup System

    Args:
        system_config (SystemConfig): return configuration of system setting
    """    
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
