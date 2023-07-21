# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:28:06 2023

@author: jalob
"""

"""
Normal_100x: 89 (17 %)
Cancer_100x: 439 (83 %)
Normal_400x: 201 (29 %)
Cancer_400x: 495 (71 %)

"""
from PIL import Image
import glob
import numpy as np
from torchvision import transforms
from pathlib import Path
from matplotlib import pyplot as plt
import os
import re

#################################
# DATA AUGMENTATION
##################################

# Create transform function to generate new images by DataAugmentation
data_augmentation = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(90,90)),
    ], 0.5),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(180,180)),
    ], 0.5),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(270,270)),
    ], 0.5),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(360,360)),
    ], 0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
    ])

root = "dataset/wsi_100x"

class_list = [class_name.name for class_name in list(os.scandir(root))]

class_dict = {class_name:len(list(os.scandir(root + "/" + class_name)))for class_name in class_list}

# Find the minimum value in the list

min_class = min(class_dict, key=lambda i: class_dict[i])
min_len = class_dict[min_class]

max_class = max(class_dict, key=lambda i: class_dict[i])
max_len = class_dict[max_class]

print("The underrepresented class is", min_class, "with a size of", min_len)
print("The overrepresented class is", max_class, "with a size of", max_len)

samples_path_list = glob.glob(root + "/" + min_class + "/*")
increase_size = 2
increment = 0

for i in range(increase_size):
    
    for sample_idx, sample_path in enumerate(samples_path_list):
        
        original_sample_name = os.path.basename(sample_path)
        
        original_sample_name = re.search(r"(.+)\.jpg", original_sample_name).group(1)
        # Open an existing patch using the patch index 
        sample_original = Image.open(sample_path)
        
        # Apply DataAugmentation to the original patch to create a new one
        sample_new = data_augmentation(sample_original)
        
        # Create a numpy array to print the patch 
        sample_new_array = np.asarray(sample_new)
        
        sample_name = original_sample_name + "_t_" + str(sample_idx + 1 + increment)
        # Save patch
        plt.imsave(root + "/" + min_class + "/" + sample_name + ".jpg", sample_new_array)
        print("Sample:", sample_name)
        
    increment += sample_idx + 1
    
        