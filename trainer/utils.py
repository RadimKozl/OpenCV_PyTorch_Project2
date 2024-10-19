#!/usr/bin/python3

"""Utils module

Implements helper functions.

"""

# Import libraries
import os
import random
import numpy as np

import torch
import torch.nn.functional as F

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


def prediction(model, device, batch_input, max_prob=True):
    """
    get prediction for batch inputs
    """
    
    # send model to cpu/cuda according to your system configuration
    model.to(device)
    
    # it is important to do model.eval() before prediction
    model.eval()

    data = batch_input.to(device)

    output = model(data)

    # get probability score using softmax
    prob = F.softmax(output, dim=1)
    
    if max_prob:
        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]
    else:
        pred_prob = prob.data
    
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()


def get_target_and_prob(model, dataloader, device):
    """
    get targets and prediction probabilities
    """
    
    pred_prob = []
    targets = []
    
    for _, (data, target) in enumerate(dataloader):
        
        _, prob = prediction(model, device, data, max_prob=False)
        
        pred_prob.append(prob)
        
        target = target.numpy()
        targets.append(target)
        
    targets = np.concatenate(targets)
    targets = targets.astype(int)
    pred_prob = np.concatenate(pred_prob, axis=0)
    
    return targets, pred_prob


def get_target_and_classes_cm(model, dataloader, device):
    """
    Get true targets and predicted classes from the model.
    """
    model.eval()
    targets = []
    pred_classes = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            # Get model output (logits or probabilities)
            output = model(data)
            
            # Get predicted classes (use argmax to get the index of the highest probability)
            pred = torch.argmax(output, dim=1)
            
            # Append to lists
            pred_classes.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    # Convert lists to numpy arrays
    targets = np.concatenate(targets)
    pred_classes = np.concatenate(pred_classes)
    
    return targets, pred_classes


def save_model(model, device, model_dir='models', model_file_name='model.pt'):
    """Function of save model

    Args:
        model (torch.nn.Module): torch model for save
        device (torch.device): setting type of calculation device CPU/GPU. Defaults to "cuda"
        model_dir (str, optional): save directory. Defaults to 'models'.
        model_file_name (str, optional): file name of model. Defaults to 'model.pt'.
    """    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    
    if device == 'cuda':
        model.to('cuda')
    
    return


def load_model(model, model_dir='models', model_file_name='model.pt', weights_only=False):
    """Function for load model

    Args:
        model (torch.nn.Module): torch model for save
        model_dir (str, optional): save directory. Defaults to 'models'.
        model_file_name (str, optional): file name of model. Defaults to 'model.pt'.
        weights_only (bool, optional): value for setting load weights of model. Defaults to False.

    Returns:
        torch.nn.Module: return load model
    """    
    model_path = os.path.join(model_dir, model_file_name)

    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path, weights_only=weights_only))
    
    return model