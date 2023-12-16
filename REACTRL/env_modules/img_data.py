# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split


import cv2
import glob
from roboflow import Roboflow
import pandas as pd

# Define dataset class

class diss_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, img_channel=1, label_path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/image-quality-and-molecule-type-1/train'):
        # X is dissassemble data, here choose topography as input, y is the label to check if dissociation happens
        self.dataset_list = dataset_list
        self.label_path=label_path
        self.img_channel=img_channel
        
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx, multi_label=False):
        if self.img_channel==1:
            x=cv2.imread(self.dataset_list[idx], cv2.IMREAD_GRAYSCALE)
        elif self.img_channel==3:
            x=cv2.imread(self.dataset_list[idx])
        x=cv2.resize(x, (128, 128), interpolation = cv2.INTER_AREA)  # resize to 128*128
        if multi_label:
            if 'good' in self.dataset_list[idx]:
                y=np.array([1, 0, 0])
                # y=0
            elif 'bad' in self.dataset_list[idx]:
                y=np.array([0, 1, 0])
                # y=1
            else:
                y=np.array([0, 0, 1])
                # y=2
            return x, y
        else:
            label_all=pd.read_csv(os.path.join(self.label_path, '_classes.csv'))
            y=label_all[label_all['filename']==self.dataset_list[idx].split('/')[-1]][' true_mol'].values[0]
            if y==1:
                y=np.array([1, 0])
            else:
                y=np.array([0, 1])
    
            return x, y
        
def get_dataset_list(dataset_path='image-quality-and-molecule-type-1', test_dataset_path=False, val_dataset_path=False):

# Define dataset class
# dataset_path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/image-quality-and-molecule-type-1'
    if os.path.exists(dataset_path):
        print("Dataset already downloaded")
    else:  #  download from roboflow
        print("Downloading dataset")
        rf = Roboflow(api_key="ffgrnIQXcbWXDXC3xOAA")
        project = rf.workspace("aaltouniversity").project("scan-image-quality")
        dataset = project.version(2).download("folder")
        dataset_path = dataset.location

    # Obtain list of images
    dataset_list = glob.glob(os.path.join(dataset_path, 'train', '*.jpg'), recursive = True)
    try:
        test_dataset_list = glob.glob(os.path.join(dataset_path, 'test', '*.jpg'), recursive = True)
        val_dataset_list = glob.glob(os.path.join(dataset_path, 'valid', '*.jpg'), recursive = True)
        return dataset_list, test_dataset_list, val_dataset_list
    except:
        return dataset_list


# Define dataset loader
def get_dataset(dataset_list, img_channel=1, test_dataset_list=None, val_dataset_list=None, batch_size=12, test_size=0.2, shuffle=True):
    dataset=diss_Dataset(dataset_list, img_channel=img_channel)
    if test_dataset_list is not None:
        train_dataset=dataset
        test_dataset=diss_Dataset(test_dataset_list, img_channel=img_channel, label_path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/image-quality-and-molecule-type-1/test')
    else:
        train_dataset, test_dataset=train_test_split(dataset, test_size=test_size, random_state=42, shuffle=shuffle)
    
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    if val_dataset_list is not None:
        val_dataset=diss_Dataset(val_dataset_list, img_channel=img_channel, label_path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/image-quality-and-molecule-type-1/valid')
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset
    else:
        return train_loader, test_loader, train_dataset, test_dataset