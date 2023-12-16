
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
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



class img_classifier(nn.Module):
    def __init__(self, model, train_loader=None, val_loader=None, output_dir='output', tensorboard=True, epochs=200, optimizer='Adam', loss_fn='cross_entropy_loss', device="cpu"):
        super(img_classifier, self).__init__()


        self.model = self.model_to_device(model, device)


        if optimizer=='Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        elif optimizer=='SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        if loss_fn=='cross_entropy_loss':
            self.loss_fn = nn.CrossEntropyLoss()

        if tensorboard:
            self.tensorboard_writer = SummaryWriter('runs/')


        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir



        with open(os.path.join(output_dir, "train_output_batch.txt"), "a") as train_output_batch:
            train_output_batch.write("Training output batch\n")
        self.train_output_batch = train_output_batch

        with open(os.path.join(output_dir, "train_output_episode.txt"), "a") as train_output_episode:
            train_output_episode.write("Training output episode\n")
        self.train_output_episode = train_output_episode
        
        with open(os.path.join(output_dir, "val_output_batch.txt"), "a") as val_output_batch:
            val_output_batch.write("Validation output batch\n")
        self.val_output_batch = val_output_batch
        with open(os.path.join(output_dir, "val_output_episode.txt"), "a") as val_output_episode:
            val_output_episode.write("Validation output episode\n")
        self.val_output_episode = val_output_episode

        with open(os.path.join(output_dir, "test_output_batch.txt"), "a") as test_output_batch:
            test_output_batch.write("Test output batch\n")
        self.test_output_batch = test_output_batch
        with open(os.path.join(output_dir, "test_output_episode.txt"), "a") as test_output_episode:
            test_output_episode.write("Test output episode\n")
        self.test_output_episode = test_output_episode
    

    def model_train(self,
                    optimizer, 
                    loss_fn, 
                    train_loader, 
                    val_loader, 
                    epochs, 
                    device="cpu",
                    tensorboard=False,):
        """
        Trains the model on the given dataset for the specified number of epochs. 
        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be trained
        optimizer: torch.optim.Optimizer
            The optimizer to be used for training
        loss_fn: torch.nn.modules.loss
            The loss function to be used for training
        train_loader: torch.utils.data.DataLoader
            The training dataset
        val_loader: torch.utils.data.DataLoader
            The validation dataset
        epochs: int
            The number of epochs for which the model is to be trained
        device: str
            The device on which the model is to be trained
        Returns
        -------
        model: torch.nn.Module
            The trained model
        train_losses: list
            The training losses for each epoch
        val_losses: list
            The validation losses for each epoch
        """


        loss_min=1000
        for epoch in range(epochs):
            self.model.train()
            acc_episode=[]
            loss_episode=[]
            for i, data in enumerate(train_loader):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                acc=(torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(targets, dim=1), dim = 1).indices).sum()
                loss.backward()
                optimizer.step()
                loss_episode.append(loss.item())
                acc_episode.append(acc.item())
                self.train_output_batch.write(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")
                if tensorboard:
                    self.tensorboard_writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader) + i)
                    self.tensorboard_writer.add_scalar('Accuracy/train', acc.item(), epoch*len(train_loader) + i)

            
            loss_episode_avg = np.mean(loss_episode)
            acc_episode = np.sum(acc_episode)
            if tensorboard:
                self.train_output_episode.write(f"Epoch: {epoch}, Episode_loss: {loss_episode_avg}, Episode_accuracy: {acc_episode}\n")

            if epoch%10==0:
                # validation loss
                val_loss, val_acc = self.model_test(val_loader, mode='val', device=device)
                self.val_output_batch.write(f"Epoch: {epoch}, Loss: {val_loss}, Accuracy: {val_acc}\n")
                if tensorboard:
                    self.tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
                    self.tensorboard_writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.model_save(os.path.join(self.output_dir, f"model_{epoch}.pth"))
                if val_loss<loss_min:
                    loss_min=val_loss
                    self.model_save(os.path.join(self.output_dir, "model_best.pth"))

    


    
    def model_test(self,
                test_loader,
                mode="test", 
                device="cpu"):
        """
        Tests the model on the given dataset. 
        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be trained
        test_loader: torch.utils.data.DataLoader
            The test dataset
        device: str
            The device on which the model is to be trained
        Returns
        -------
        test_loss: float
            The test loss
        test_accuracy: float
            The test accuracy
        """
        self.model.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                acc=(torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(targets, dim=1), dim = 1).indices).sum()
                test_loss += loss.item()
                test_accuracy += acc.item()
                if mode=="test":
                    self.test_output_batch.write(f"Batch: {i}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")
                elif mode=="val":
                    self.val_output_batch.write(f"Batch: {i}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        return test_loss, test_accuracy
    
    def model_save(self, 
                path):
        """
        Saves the model at the specified path
        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be saved
        optimizer: torch.optim.Optimizer
            The optimizer to be saved
        path: str
            The path at which the model is to be saved
        Returns
        -------
        None
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        
    def model_load(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path))


    def calculate_auc_roc(self, preds, targets, pos_label=None):
        fpr, tpr, thresh = roc_curve(preds, targets, pos_label=pos_label)
        auc_score = roc_auc_score(preds, targets)
        return fpr, tpr, thresh, auc_score
    
    def calculate_accuracy(self, preds, targets):
        return (torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(targets, dim=1), dim = 1).indices).sum()
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def data_to_device(self, data, device):
        if isinstance(data, (list,tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    
    def model_to_device(self, device):
        for layer in self.model.children():
            layer.to(device)

