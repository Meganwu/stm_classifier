
import numpy as np
from data_path import data_path
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import glob

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from REACTRL.env_modules.net_framework_current import img_classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from boss.bo.bo_main import BOMain
from boss.pp.pp_main import PPMain

def plot_diss_results(img_path, episodes=5, diss_times=10, col=4, figsize=(15,600)):
    diss_before_dir=os.path.join(img_path, 'diss_before_img')
    diss_data_dir=os.path.join(img_path, 'diss_data')
    diss_after_dir=os.path.join(img_path, 'diss_after_img')
    sample_num=len(os.listdir(diss_before_dir))
    succ=0
    plt.figure(figsize=figsize)
    with open('label.txt', 'a') as f:
        f.write('episode, diss_i, done_diss, before, after\n')
    for i in range(episodes):
        for j in range(diss_times):
            if os.path.exists(os.path.join(diss_before_dir, 'diss_before_{}_{}.png'.format(i,j))):
                img_before=cv2.imread(os.path.join(diss_before_dir, 'diss_before_{}_{}.png'.format(i,j)))
                diss_data=pickle.load(open(os.path.join(diss_data_dir, 'vert_data_{}_{}.pkl'.format(i,j)), 'rb'))
                img_after=cv2.imread(os.path.join(diss_after_dir, 'diss_after_{}_{}.png'.format(i,j)))
                plt.subplot(sample_num, col, succ*col+1)
                plt.imshow(img_before)
                plt.ylabel('before %s_%s'%(i,j), fontsize='large')
                
                plt.subplot(sample_num, col, succ*col+2)
                # plt.plot(diss_data.V, diss_data.topography)
                topography=np.array(diss_data.topography)
                voltage=np.array(diss_data.V)
                diff_topography=np.abs(topography[0:512].sum()-topography[512:].sum())
                if voltage.max()>2.5 and diff_topography>2.0:
                    with open('label.txt', 'a') as f:
                        f.write('%s, %s, 1, 1, 1\n')
                    plt.plot(diss_data.V, diss_data.topography, 'r') 
                else:
                    with open('label.txt', 'a') as f:
                        f.write('%s, %s, 0, 1, 1\n')
                    plt.plot(diss_data.V, diss_data.topography, 'b')
                plt.ylabel('%s' % diff_topography, fontsize='large')
                
                plt.subplot(sample_num, col, succ*col+3)
                plt.plot(diss_data.V, diss_data.current)
                
                plt.subplot(sample_num, col, succ*col+4)
                plt.imshow(img_after)
                plt.ylabel('after %s_%s'%(i,j), fontsize='large')
                succ+=1
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

 
class current_Dataset():
    def __init__(self, data_path):
        self.data_path=data_path
        self.data_list=glob.glob(os.path.join(data_path, '*/diss_data/*'))
        self.len=len(self.data_list)
        
    def __getitem__(self, index, plot=True):
        data=self.data_list[index]
        diss_data=pickle.load(open(data, 'rb'))
        topography=np.array(diss_data.topography)
        current=np.array(diss_data.current)
        voltage=np.array(diss_data.V)
        diff_topography=np.abs(topography[0:512].sum()-topography[512:].sum())
        if voltage.max()>2.5 and diff_topography>2.0:
            label=[1, 0]
        else:
            label=[0, 1]
        if plot:
            if label==[1, 0]:
                plt.plot(diss_data.V, diss_data.topography, 'r')
            else:
                plt.plot(diss_data.V, diss_data.topography, 'b')
            plt.ylabel('%.2f' % diff_topography, fontsize='large')
            
        label=np.array(label)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        current = scaler.fit_transform(current.reshape(-1, 1))
        return current, label

        # return topography, label
    
    def __len__(self):
        return self.len
    
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler
    
class CONV(nn.Module):
    def __init__(self,input_dim, kernel_size, max_pool_kernel_size, stride, max_pool_stride, output_dim):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout1 = self.get_size(input_dim, kernel_size, stride=stride)
        self.max_pool1 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout1_1 = self.get_size(lout1, max_pool_kernel_size, stride=max_pool_stride)
        self.conv2 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout2 = self.get_size(lout1_1, kernel_size, stride=stride)
        self.max_pool2 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout2_1 = int(self.get_size(lout2, max_pool_kernel_size, stride=max_pool_stride))
        self.fc3 = nn.Linear(lout2_1, output_dim)
        self.dropout= nn.Dropout(0.1)
        self.float()
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.dropout(x)
        x= torch.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        # x = F.softmax(self.fc3(x), dim=1)
        return x.squeeze()
    def get_size(self, Lin, kernel_size, stride = 1, padding = 0, dilation = 1):
        Lout = (Lin + 2*padding - dilation*(kernel_size-1)-1)/stride + 1
        return Lout


class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TimeSeriesClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Extract the output from the last time step
        lstm_last_out = lstm_out[:, -1, :]
        
        # Fully connected layer for classification
        output = self.fc(lstm_last_out)
        
        return output
    
class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
    
    
    

# Assuming univariate time series data with one feature
input_size = 1
hidden_size = 50
num_classes = 2  # Adjust based on your classification task

# Instantiate the model
# model = TimeSeriesClassifier(input_size, hidden_size, num_classes)

# model = CONV(1024, 64, 4, 4, 2, 2)
model = LSTMClassifier(1, 256, 3, 2)

model.load_state_dict(torch.load('model_best.pth'))
    


data_path = data_path
img_paths = os.listdir(data_path)

dataset=current_Dataset(data_path)
shuffle=True
batch_size=12
train_dataset, test_dataset=train_test_split(dataset, test_size=0.2, random_state=42, shuffle=shuffle)
test_dataset, val_dataset=train_test_split(test_dataset, test_size=0.5, random_state=42, shuffle=shuffle)

train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=shuffle, num_workers=0)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# lr=0.001
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# scheduler = CyclicLR(optimizer, cosine(t_max=100 * 2, eta_min=lr/100))

def func(X):
    lr = X[0, 0]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # scheduler = CyclicLR(optimizer, cosine(t_max=100 * 2, eta_min=lr/100))
    # scheduler = CyclicLR(optimizer, cosine(t_max=100 * 2, eta_min=lr/100))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    current_time=datetime.now()
    output_dir='output_%s_%s%s%s' % (lr, current_time.month, current_time.hour, current_time.minute)
    net=img_classifier(model, device='cuda', output_dir=output_dir)
    net.model_train(train_loader=train_loader, val_loader=val_loader, episodes=150, device='cuda', tensorboard=True,optimizer=optimizer,
                    scheduler=scheduler)
    data=pd.read_csv(os.path.join(output_dir, 'train_output_episode.txt'), sep=',')
    acc=data[' Accuracy'].max()
    
    return acc

bounds = np.array([[0.00005, 0.005]])

if __name__=='__main__':
    lr = 8.966E-05
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # scheduler = CyclicLR(optimizer, cosine(t_max=100 * 2, eta_min=lr/100))
    # scheduler = CyclicLR(optimizer, cosine(t_max=100 * 2, eta_min=lr/100))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    current_time=datetime.now()
    output_dir='output_%s_%s%s%s' % (lr, current_time.day, current_time.hour, current_time.minute)
    net=img_classifier(model, device='cuda', output_dir=output_dir)
    net.model_train(train_loader=train_loader, val_loader=val_loader, episodes=2000, device='cuda', tensorboard=True,optimizer=optimizer,
                    scheduler=scheduler) 
