# -*- coding: utf-8 -*-
"""vgg_wsi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SX1vye963NKKBpuFyzNYnzh_EPuxpdCk

from google.colab import drive
drive.mount('/content/drive')
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import glob
import os
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import torchmetrics
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, Subset
import pickle
import torchvision.models as models


from argparse import ArgumentParser
parser = ArgumentParser(description="CNN_MIL")
parser.add_argument("-r", action="store", dest="root", type=str ,help="Root of the sample folders")
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f">> Using device: {device}")

# 1. Subclass torch.utils.data.Dataset
class Dataset_Custom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, root):

        # 1. SAMPLE PATH LIST
        root_all_samples = root + "/*"

        self.device = device
        self.sample_path_list = glob.glob(root_all_samples)
        self.labels = [torch.load(sample)[1] for sample in self.sample_path_list]

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self):
        "Returns the total number of samples."
        return len(self.sample_path_list)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, sample_idx):
        "Returns one sample of data, data and label (X, y)."

        sample_path = self.sample_path_list[sample_idx]
        input = torch.load(sample_path)

        x_cnn_sample = input[0]
        y_ffnn_sample = input[1]

        return x_cnn_sample, y_ffnn_sample # return data, label (X, y)

#########################################################
# CREATION OF DATASET
#########################################################

#root = "/content/drive/MyDrive/CNN_MIL/input_tensors_dataaug"
root = args.root

dataset = Dataset_Custom(root)

labels = torch.stack(dataset.labels)

train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify = labels, random_state=2)

train_dataset = Subset(dataset, indices = train_idx)

test_dataset = Subset(dataset, indices = test_idx)

#########################################################
# CREATION OF DATALOADER
#########################################################

labels_train = labels[train_idx]

labels_train = labels_train.argmax(dim = 1)

class_counts = torch.bincount(labels_train)

# Calculate the sample weights
weights = 1.0 / class_counts[labels_train]

# Create the WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))

batch_size = 8

# Divide the dataset into batches
dataloader_train = DataLoader(dataset = train_dataset, batch_size = batch_size, sampler = sampler)
dataloader_test = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)

class vgg_MIL(nn.Module):

    def __init__(self, vgg, n_output):
        super(vgg_MIL, self).__init__()

        # Incorporta vgg
        self.cnn = vgg.features
        self.avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.ffnn = vgg.classifier

        # Freeze the parameters of the feature extraction part
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Modify the classifier step (last layer)
        self.ffnn[6] = nn.Linear(in_features=4096, out_features=n_output, bias=True)

    def forward(self, x_cnn):

        # Get the final output for each batch from the feed forward part
        y_ffnn = self.ffnn(self.flatten(self.avgpool(self.cnn(x_cnn))))

        return y_ffnn

n_output = train_dataset[0][1].size(0)

# Load the pre-trained VGG model
vgg = models.vgg16(pretrained=True)

# Create an instance of the modified VGG model
model = vgg_MIL(vgg, n_output).to(device)

learning_rate = 0.1

# Create an optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train_test_epoch(dataloader_train, dataloader_test, model, epochs):

  loss_train_list = list()
  loss_test_list = list()

  acc_train_list = list()
  acc_test_list = list()

  prec_train_list = list()
  prec_test_list = list()

  rec_train_list = list()
  rec_test_list = list()

  spec_train_list = list()
  spec_test_list = list()

  for epoch_idx in range(epochs):

    #########################################################
    # TRAIN LOOP
    #########################################################

    batch_loss_train = 0
    y_ffnn_pred_label_list = list()
    y_ffnn_target_label_list = list()

    model.train()

    for x_cnn_batch, y_ffnn_batch_target in dataloader_train:

      #########################################################
      # GET OUTPUT PREDICTED
      #########################################################

      # Get the predicted output "y_ffnn_batch_pred" for each batch
      y_ffnn_batch_pred = model(x_cnn_batch.to(device))

      #########################################################
      # LOSS
      #########################################################

      # Calculate loss
      loss = criterion(y_ffnn_batch_pred, y_ffnn_batch_target.to(device))
      batch_loss_train += loss.data

      # Apply backpropagation to update parameters (weights and bias)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      #########################################################
      # STORE OUTPUT
      #########################################################

      y_ffnn_batch_pred_label = y_ffnn_batch_pred.argmax(dim = 1)
      y_ffnn_batch_target_label = y_ffnn_batch_target.argmax(dim = 1)

      y_ffnn_pred_label_list.append(y_ffnn_batch_pred_label)
      y_ffnn_target_label_list.append(y_ffnn_batch_target_label)

    #########################################################
    # ACCURACY
    #########################################################

    y_ffnn_pred_label = torch.cat(y_ffnn_pred_label_list, dim = 0).cpu().numpy()
    y_ffnn_target_label = torch.cat(y_ffnn_target_label_list, dim = 0).cpu().numpy()

    acc_train, prec_train, rec_train, spec_train = performance_metrics(y_ffnn_pred_label, y_ffnn_target_label)

    ########################################################
    ########################################################
    ########################################################
    ########################################################

    #########################################################
    # TEST LOOP
    #########################################################

    batch_loss_test = 0
    y_ffnn_pred_label_list = list()
    y_ffnn_target_label_list = list()

    model.eval()

    with torch.inference_mode():

      for x_cnn_batch, y_ffnn_batch_target in dataloader_test:

        #########################################################
        # GET OUTPUT PREDICTED
        #########################################################

        # Get the predicted output "y_ffnn_batch_pred" for each batch
        y_ffnn_batch_pred = model(x_cnn_batch.to(device))

        #########################################################
        # LOSS
        #########################################################

        # Calculate loss
        loss = criterion(y_ffnn_batch_pred, y_ffnn_batch_target.to(device))
        batch_loss_test += loss.data

        #########################################################
        # STORE OUTPUT
        #########################################################

        y_ffnn_batch_pred_label = y_ffnn_batch_pred.argmax(dim = 1)
        y_ffnn_batch_target_label = y_ffnn_batch_target.argmax(dim = 1)

        y_ffnn_pred_label_list.append(y_ffnn_batch_pred_label)
        y_ffnn_target_label_list.append(y_ffnn_batch_target_label)

      #########################################################
      # ACCURACY
      #########################################################

      y_ffnn_pred_label = torch.cat(y_ffnn_pred_label_list, dim = 0).cpu().numpy()
      y_ffnn_target_label = torch.cat(y_ffnn_target_label_list, dim = 0).cpu().numpy()

      acc_test, prec_test, rec_test, spec_test = performance_metrics(y_ffnn_pred_label, y_ffnn_target_label)

    #########################################################
    # STORE LOSSES AND ACC
    #########################################################

    # Store the train and test loss for each epoch

    loss_train = float(batch_loss_train / len(dataloader_train))
    loss_test = float(batch_loss_test / len(dataloader_test))

    loss_train_list.append(loss_train)
    loss_test_list.append(loss_test)

    # Store the train and test performance metrics for each epoch

    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)

    prec_train_list.append(prec_train)
    prec_test_list.append(prec_test)

    rec_train_list.append(rec_train)
    rec_test_list.append(rec_test)

    spec_train_list.append(spec_train)
    spec_test_list.append(spec_test)

    print("Epoch:", epoch_idx + 1)
    print("Train loss:", loss_train)
    print("Test loss:", loss_test)
    print("Train accuracy:", acc_train)
    print("Test accuracy:", acc_test)
    print("Train precision:", prec_train)
    print("Test precision:", prec_test)
    print("Train recall:", rec_train)
    print("Test recall:", rec_test)
    print("Train specificity:", spec_train)
    print("Test specificity:", spec_test)

  return model, loss_train_list, loss_test_list, acc_train_list, acc_test_list, prec_train_list, prec_test_list, rec_train_list, rec_test_list, spec_train_list, spec_test_list

def performance_metrics(pred, target):

  # Calculate TP, TN, FP, FN
  TP = np.sum(np.logical_and(pred == 1, target == 1))
  TN = np.sum(np.logical_and(pred == 0, target == 0))
  FP = np.sum(np.logical_and(pred == 1, target == 0))
  FN = np.sum(np.logical_and(pred == 0, target == 1))

  # Calculate metrics
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  specificity = TN / (TN + FP)

  return accuracy, precision, recall, specificity

def plot_function(train_metric, test_metric, metric):

    plt.figure(figsize=(15,4))
    plt.plot(list(range(1, len(train_metric) + 1)), train_metric, label='Training ' + metric)
    plt.plot(list(range(1, len(test_metric) + 1)), test_metric, label='Testing ' + metric)

    # Select range x axis
    # plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

    # find position of lowest test loss
    minposs = test_metric.index(min(test_metric)) + 1
    plt.axvline(minposs, linestyle='--', color='r',label='Minimum Test ' + metric)

    plt.title("Plot " + metric + " of each epoch")
    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.show()

epochs = 100

model, train_loss, test_loss, train_acc, test_acc, train_prec, test_prec, train_rec, test_rec, train_spec, test_spec = train_test_epoch(dataloader_train, dataloader_test, model, epochs)

# Create a dictionary to store tensors and the model
data_dict = {
    "train_loss": train_loss,
    "test_loss": test_loss,
    "train_acc": train_acc,
    "test_acc": test_acc,
    "train_prec": train_prec,
    "test_prec": test_prec,
    "train_rec": train_rec,
    "test_rec": test_rec,
    "train_spec": train_spec,
    "test_spec": test_spec,
    "model": model.state_dict()
}

# Save the dictionary in a .pth file
torch.save(data_dict, 'vgg_wsi.pth')

print("Train loss:", train_loss)
print("Test loss:", test_loss)
print("Train accuracy:", train_loss)
print("Test accuracy:", test_loss)
print("Train precision:", train_prec)
print("Test precision:", test_prec)
print("Train recall:", train_rec)
print("Test recall:", test_rec)
print("Train specificity:", train_spec)
print("Test specificity:", test_spec)

plot_function(train_loss, test_loss, "Loss")
plot_function(train_acc, test_acc, "Accuracy")
plot_function(train_prec, test_prec, "Precision")
plot_function(train_rec, test_rec, "Recall")
plot_function(train_spec, test_spec, "Specificity")