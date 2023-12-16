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
    def __init__(self, dataset_list, path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/REACTRL/task_script/image-quality-and-molecule-type-1'):
        # X is dissassemble data, here choose topography as input, y is the label to check if dissociation happens
        self.dataset_list = dataset_list
        self.path=path
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx, single_label=True):
        x=cv2.imread(self.dataset_list[idx], cv2.IMREAD_GRAYSCALE)
        x=cv2.resize(x, (128, 128), interpolation = cv2.INTER_AREA)  # resize to 128*128
        if single_label:
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
            label_all=pd.read_csv(os.path.join(slf.path, 'train', '_classes.csv'))
            y=label_all[label_all['filename']==self.dataset_list[idx].split('/')[-1]][' true_mol'].values[0]
            if y==1:
                y=np.array([1, 0])
            else:
                y=np.array([0, 1])
    
            return x, y


# Define dataset class
dataset_path='C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/reaction_rl/REACTRL/task_script/image-quality-and-molecule-type-1'
if os.path.exists(dataset_path):
    print("Dataset already downloaded")
else:
    print("Downloading dataset")
    rf = Roboflow(api_key="ffgrnIQXcbWXDXC3xOAA")
    project = rf.workspace("aaltouniversity").project("scan-image-quality")
    dataset = project.version(2).download("folder")
    dataset_path = dataset.location


# Obtain list of images
dataset_list = glob.glob(os.path.join(dataset_path, 'train', '*.jpg'), recursive = True)


# Define dataset loader

dataset=diss_Dataset(dataset_list)
train_dataset, test_dataset=train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
device=torch.cuda.is_available()
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=0)