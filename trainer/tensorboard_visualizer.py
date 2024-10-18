#!/usr/bin/python3

"""TensorBoard Visualizer Class module
"""

# Import libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any

from .visualizer import Visualizer
from .utils import get_target_and_prob


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


class ModelVisualizer:
    """Class of visualize graph of model inside TensorBoard

    Args:
        LogSetting (class): Abstract class of LogSetting Base class
    """    
    def __init__(self, model, inputs, writer):
        """Init method of class

        Args:
            model (torch.nn.Module): model definition
            inputs (torch.utils.data.DataLoader): inputs for models
            writer (obj): summary writer from PyTorch
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
        

class DataEmbedingVisualizer:
    """Class of Data Embeding for TensorBoard

    Args:
        LogSetting (class): Abstract class of LogSetting Base class
    """    
    def __init__(self, dataset, writer, number_samples=100, num_workers=2, shuffle=True, tag="embedings"):
        """Init method of class

        Args:
            dataset (obj): return data of samples and data of labels
            class_labels (list): list of class labels names
            writer (obj): summary writer from PyTorch
            number_samples (int, optional): number selected samples. Defaults to 100.
            global_step (int, optional): number of step. Defaults to 1.
            tag (str, optional): tag of destription. Defaults to "embedings".
        """        
        super().__init__()
        self.writer = writer
        self.number_samples = number_samples
        self.tag = tag
        self.dataset = dataset
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def update_charts(self):
        """
        Add a few inputs and labels to tensorboard. 
        """
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.number_samples, num_workers=self.num_workers, shuffle=self.shuffle)
        
        images, labels = next(iter(dataloader))
        
        
        # Add image as embedding to tensorboard
        self.writer.add_embedding(mat = images.view(-1, 3 * 224 * 224), 
                                metadata=labels,
                                label_img=images,
                                tag=self.tag)
        return

    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self.writer.close()
        
class WeightsHistogramVisualizer:
    """Class for adding histograms of weights
    """    
    def __init__(self, writer):
        """Init method of class

        Args:
            writer (obj): summary writer from PyTorch
            
        """
        super().__init__()
        self.writer = writer
        
    def update_charts(self, model, epoch):
        """Metod for adding histogram to TenroBoard

        Args:
            model (torch.nn.Module): trained model
            epoch (int): epoch number
        """        
        
        for name, param in model.named_parameters():
            self.writer.add_histogram(name.replace('.', '/'), param.data.cpu().abs(), epoch)
            
        return
    
    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self.writer.close()

    
class PRVisualizer:
    """Class for adding PR curve to the TensorBoard
    """    
    def __init__(self, dataloader, writer, name_classes):
        """Init method of class

        Args:
            dataloader (torch.utils.data.DataLoader): validation dataset loader
            writer (obj): summary writer from PyTorch
            name_classes (list): list of names classes
        """        
        self.dataloader = dataloader
        self.writer = writer
        self.name_classes = name_classes
        self.num_classes = len(self.name_classes)
    
    def update_charts(self, model, device, epoch):
        """Method for adding PR curve

        Args:
            model (torch.nn.Module): torch model to validate
            device (torch.device): setting type of calculation device CPU/GPU. Defaults to "cuda".
            epoch (int): epoch number
        """        
        targets, pred_prob = get_target_and_prob(model, self.dataloader, device)
        
        for cls_idx in range(self.num_classes):
            binary_target = targets == cls_idx
            true_prediction_prob = pred_prob[:, cls_idx]
        
            self.writer.add_pr_curve(self.name_classes[cls_idx], 
                                binary_target, 
                                true_prediction_prob, 
                                global_step=epoch)
        
    
    def close_tensorboard(self):
        """Close method of class, close defined SummaryWriter()
        """        
        self.writer.close()
        