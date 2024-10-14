#!/usr/bin/python3

"""Module custom dataloader

This module store class for create PyTorch Dataloader from JSON file
"""

# Import libraries
import json

from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class JsonClassificationDataset(Dataset):
    """Class for create Dataset for PyTorch from JSON file data

    Args:
        Dataset (class): PyTorch Dataset class
    """    
    def __init__(self, json_file, type_data='train', dataset_number=0, image_shape=None, transform=None):
        """Init method of class

        Args:
            json_file (str): path of JSON file with data of datasets
            type_data (str, optional): Setting of load type of dataset - 'train'/'valid'/'test' data. Defaults to 'train'.
            dataset_number (int, optional): This is number of creted variant of datasets. Defaults to 0.
            image_shape (int/tuple, optional): value of weight & height of resized image. Defaults to None.
            transform (torchvision.transforms.Compose, optional): list of transformation of PyTorch. Defaults to None.

        """
        super().__init__()
        self.json_file = json_file
        with open(self.json_file, 'r', encoding="utf-8") as f:
            self.config_datatasets = json.load(f)

        self.dataset_number = int(dataset_number)
        self.type_data = type_data
        self.base_setting = list(self.config_datatasets['datasets'][0].keys())

        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)

            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError

        else:
            self.image_shape = image_shape

        # set transform attribute
        self.transform = transform

        self.num_classes = self.config_datatasets['datasets'][self.dataset_number]['class_number']

        # initialize the data dictionary
        self.data_dict = {
            'image_path': [],
            'label': []
        }

        self._load_dataset()

    def _load_train_dataset(self):
        """Funcion for load train dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['train'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['train'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_valid_dataset(self):
        """Function for load valid dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['valid'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['valid'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_test_dataset(self):
        """Function for load test dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['test'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['test'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_dataset(self):
        """Internal Method for load selected dataset

        Returns:
            str: if is wrong selection parameter, it was returned warring message.
        """        
        if self.type_data == 'train':
            if 'train' in self.base_setting:
                self._load_train_dataset()
            else:
                print('Json file does not contain dataset train')

        elif self.type_data == 'valid':
            if 'valid' in self.base_setting:
                self._load_valid_dataset()
            else:
                print('Json file does not contain dataset valid')

        elif self.type_data == 'test':
            if 'test' in self.base_setting:
                self._load_test_dataset()
            else:
                print('Json file does not contain dataset test')
        else:
            return 'False settings of type_data parameter.'

    def __len__(self):
        """Method of return length of the dataset
        
        Returns:
            str: if is wrong selection parameter, it was returned warring message.
        """
        if self.type_data == 'train':
            return len(self.config_datatasets['datasets'][self.dataset_number]['train'])
        elif self.type_data == 'valid':
            return len(self.config_datatasets['datasets'][self.dataset_number]['valid'])
        elif self.type_data == 'test':
            return len(self.config_datatasets['datasets'][self.dataset_number]['test'])
        else:
            return 'False settings of type_data parameter.'

    def __getitem__(self, idx):
        """Method for given index, return images with resize and preprocessing.

        Args:
            idx (int): number of index of image

        Returns:
            np.array, int: return image as np.array and number of class as int
        """        
        image = Image.open(self.data_dict['image_path'][idx]).convert("RGB")

        if self.image_shape is not None:
            image = F.resize(image, self.image_shape)

        if self.transform is not None:
            image = self.transform(image)

        target = self.data_dict['label'][idx]

        return image, target


    def common_name(self, label):
        """
        Method of class label to common name mapping
        """
        list_labels = list(self.config_datatasets['datasets'][0]['names_class'])
        return list_labels[label]
    
    def number_of_class(self):
        """Method for return number of class of datasets

        Returns:
            int: number of class of dataset
        """        
        return self.num_classes
    
    def names_of_classe(self):
        """Method return names of classes

        Returns:
            list(str): list of names of class
        """        
        return list(self.config_datatasets['datasets'][0]['names_class'])
