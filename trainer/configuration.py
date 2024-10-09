#!/usr/bin/python3

"""Configurations module
"""

# Import libraries
from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor


@dataclass
class SystemConfig:
    """Class of System Configuration
    
    Args:
        seed (int, optional): Seed number to set the state of all random number generators
        cudnn_benchmark_enabled (bool, optional): Enable CuDNN benchmark for the sake of performance
        cudnn_deterministic (bool, optional): Make cudnn deterministic (reproducible training)
    """    
    seed: int = 42
    cudnn_benchmark_enabled: bool = False
    cudnn_deterministic: bool = True
    
    
@dataclass
class DatasetConfig:
    """Class of Data Configuration
    
    Args:
        root_dir (str, optional): Dataset directory root
        train_transforms (torch.Tensor, optional): Data transformation to use during training data preparation
        test_transforms (torch.Tensor, optional): Data transformation to use during test data preparation
    """    
    root_dir: str = "data"
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )


@dataclass
class DataloaderConfig:
    """Class of Dataloader Configuration
    
    Args:
        batch_size (int, optional): Amount of data to pass through the network at each forward-backward iteration
        num_workers (int, optional): Number of concurrent processes using to prepare data, for free Colab num_workers=2, for free Kaggle num_workers=4
    """    
    batch_size: int = 250
    num_workers: int = 2


@dataclass
class OptimizerConfig:
    """Class of Optimizer Configuration
    
    Args:
        learning_rate (float, optional): Determines the speed of network's weights update
        momentum (float, optional): Used to improve vanilla SGD algorithm and provide better handling of local minimas
        weight_decay (float, optional): Amount of additional regularization on the weights values
        lr_step_milestones (Iterable, optional): At which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
        lr_gamma (float, optional): Multiplier applied to current learning rate at each of lr_ctep_milestones
    """    
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_step_milestones: Iterable = (
        30, 40
    )
    lr_gamma: float = 0.1


@dataclass
class TrainerConfig:
    """Class of Training Configuration
    
    Args:
        model_dir (str, optional): Directory to save model states
        model_saving_frequency (int, optional): Frequency of model state savings per epochs
        device (str, optional): Device to use for training.
        epoch_num (int, optional): Number of times the whole dataset will be passed through the network
        progress_bar (bool, optional): Enable progress bar visualization during train process
    """    
    model_dir: str = "checkpoints"
    model_saving_frequency: int = 1
    device: str = "cpu"
    epoch_num: int = 50
    progress_bar: bool = True
