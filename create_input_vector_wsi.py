# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:51:29 2023

@author: jalob
"""

import torch
import glob
import os
from PIL import Image
from pathlib import Path
#from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
import re

def create_x_cnn__y_ffnn(root, new_root):
    
    #####################
    # OBTAIN SAMPLE LIST
    #####################
    
    root_all_samples = root + "/*/*"
    
    samples_list = glob.glob(root_all_samples)
    #####################
    # OBTAIN CLASS DICT
    #####################
    
    class_dict = class_dict_func(root)
    
    #####################
    # CREATE X_CNN AND Y_FFNN
    #####################
    
    for sample_idx, sample_root in enumerate(samples_list):
    
      # Get the tensor sample "x_cnn_sample" that stores all the patch tensors
      x_cnn_sample = create_x_cnn_sample(sample_root)
      
      y_ffnn_sample = create_y_ffnn_sample(sample_root, class_dict)
      
      sample_tensors = [x_cnn_sample, y_ffnn_sample]
      
      # Get the sample name
      sample_name_complete = os.path.basename(sample_root)
      
      sample_name = re.search(r"(.+)\.jpg", sample_name_complete).group(1)
      
      sample_path = new_root + "/" + sample_name + ".pth"
      
      # Store x_cnn_sample
      torch.save(sample_tensors, sample_path)
      
      print("N_sample:", sample_idx + 1, "sample:", sample_path)
      print("x_cnn_sample shape:", x_cnn_sample.shape)
      print("y_ffnn_sample shape:", y_ffnn_sample.shape)

def class_dict_func(root):
    
    class_list = [class_name.name for class_name in list(os.scandir(root))]
    
    # Number of classes
    num_classes = len(class_list)
    
    # Assign one-hot encoding to each class
    class_labels = torch.eye(num_classes)
    
    class_to_idx = {class_list[idx]: labels for idx, labels in enumerate(class_labels)}
    
    return class_to_idx

def create_x_cnn_sample(root):

    """
    create_x_cnn_sample function takes a sample, stores the folder path of all
    their patches in patch_path_list, convert each patch to a tensor and build
    a final tensor sample that stores all the patch tensors
    """
    
    # Create the function that transforms a PIL image into a tensor
    transform = transforms.ToTensor()
    
    # Incorportate patch folder path and transform to a PIL image
    sample_PIL = Image.open(root)
  
    # Transform into a tensor
    x_cnn_sample_pre = transform(sample_PIL)
  
    # Remove the final dimension to get a 3d tensor
    x_cnn_sample = x_cnn_sample_pre[0:3][:][:]
    
    if int(len(x_cnn_sample[1])) == 2048:
        
        x_cnn_sample = x_cnn_sample.permute(0, 2, 1)
        
    return x_cnn_sample
  
def create_y_ffnn_sample(root, class_dict):

    path = Path(root)

    sample_class = path.parent.stem
    
    y_ffnn_sample = torch.tensor(class_dict[sample_class])
    
    return y_ffnn_sample
    
#####################
# APPLY FUNCTION
#####################

root = "dataset/wsi_100x"

new_root = "dataset/wsi_tensors_100x"
os.makedirs(new_root)

create_x_cnn__y_ffnn(root, new_root)
