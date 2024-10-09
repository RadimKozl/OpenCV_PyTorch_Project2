#!/usr/bin/python3

"""TensorBoard Visualizer Class module
"""

# Import libraries
from torch.utils.tensorboard import SummaryWriter

from .visualizer import Visualizer


class TensorBoardVisualizer(Visualizer):
    """Class of TensorBoard Visualizer

    Args:
        Visualizer (class): Abstract class of Visualizer Base class
    """
       
    def __init__(self):
        """Init method of class
        """        
        self._writer = SummaryWriter()


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
                self._writer.add_scalar("data/train_metric:{}".format(metric_key), metric_value, epoch)

        for test_metric_key, test_metric_value in test_metric.items():
            self._writer.add_scalar("data/test_metric:{}".format(test_metric_key), test_metric_value, epoch)

        if train_loss is not None:
            self._writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("data/test_loss", test_loss, epoch)

        self._writer.add_scalar("data/learning_rate", learning_rate, epoch)


    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self._writer.close()
