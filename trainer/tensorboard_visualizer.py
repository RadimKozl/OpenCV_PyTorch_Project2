#!/usr/bin/python3

"""TensorBoard Visualizer Class module
"""

# Import libraries
from torch.utils.tensorboard import SummaryWriter
from typing import Any

from .visualizer import LogSetting, Visualizer

# Define Summary writer from PyTorch
def set_writer(path: Any = None):
    """Function for set Summary writeru from PyTorch

    Args:
        path (Any, optional): path for log directory. Defaults to None.

    Returns:
        class: summary writer from PyTorch
    """    
    if path is not None:
        return SummaryWriter(path)
    else:
        return SummaryWriter()
    

class TensorBoardVisualizer(Visualizer):
    """Class of TensorBoard Visualizer

    Args:
        Visualizer (class): Abstract class of Visualizer Base class
    """
       
    def __init__(self, writer):
        """Init method of class

        Args:
            writer (class): summary writer from PyTorch
        """          
        self.writer = writer


    def update_charts(
        self,
        train_metric,
        train_loss,
        test_metric,
        test_loss,
        learning_rate,
        epoch
    ):
        """Update method

        Args:
            train_metric (Any): metric of training of model
            train_loss (Any): loss of training of model
            test_metric (Any): metrict of test of model
            test_loss (Any): loss of test of model
            learning_rate (float): learning rate number
            epoch (int): Number of epoch
        """
              
        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                self.writer.add_scalar("data/train_metric:{}".format(metric_key), metric_value, epoch)

        for test_metric_key, test_metric_value in test_metric.items():
            self.writer.add_scalar("data/test_metric:{}".format(test_metric_key), test_metric_value, epoch)

        if train_loss is not None:
            self.writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self.writer.add_scalar("data/test_loss", test_loss, epoch)

        self.writer.add_scalar("data/learning_rate", learning_rate, epoch)


    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self.writer.close()


class ModelVisualizer(LogSetting):
    """Class of visualize graph of model inside TensorBoard

    Args:
        LogSetting (class): Abstract class of LogSetting Base class
    """    
    def __init__(self, model, inputs, writer):
        """Init method of class

        Args:
            model (torch.nn.Module): model definition
            inputs (torch.utils.data.DataLoader): inputs for models
            writer (class): summary writer from PyTorch
        """        
        super().__init__() 
        self.model = model
        self.inputs = inputs
        self.writer = writer
        
    def update_charts(self):
        """Update method
        """        
        self.writer.add_graph(self.model, self.inputs)
        
    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self.writer.close()