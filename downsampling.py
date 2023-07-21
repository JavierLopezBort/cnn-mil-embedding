# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:58:48 2023

@author: jalob
"""
import random
import glob
import os
import torch
        
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

samples_path_list = glob.glob(root + "/" + max_class + "/*")
delete_file_list = random.sample(samples_path_list, max_len - min_len)

for delete_file in delete_file_list:
    os.remove(delete_file)
    print("File removed:", delete_file)