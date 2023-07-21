# cnn-mil-embedding

Classification of Whole Slide Images from HNSCC using CNN-MIL embedding-based predicting models

INSTALLATION REQUIREMENTS:

Required libraries:
- Openslide
- Pytorch
- Sklearn
- Path

Optional but highly recommended libraries:
- Pytorch CUDA

SCRIPTS

There are two main parts in this repository: dataset creation and feature extraction / image classification

Dataset creation
All the necessary scripts for this part are inside the "dataset" folder. You can follow these steps:

1.1 Create a root folder where you can store all the raw WSI. In our case is called "wsi_100x". Here, you should create two subfolders with the corresponding name of the classes and store the raw images inside them.

1.2 Now you can oversample if necessary. In this case, the "data_augmentation.py" file creates the new WSI images and store them in the same root folder. You can decide the number of times that you want to oversample with the variable "increase_size".

1.3 Now you can downsample if necessary. In this case, the "downsampling.py" selectes randomly a specific number of WSI from the majority class and deletes them to have the exact number of cases in both classes. Again, this is done in the original root folder.

1.4 To apply a MIL model, you need first to divide the WSI into patches. This is done by the "openslide.py" file, which creates a folder called "patches_100x" where it stores all the patches from a WSI into a folder with the sample name. It has the same folder structure as "wsi_100x".

1.5 Finally, you create the input tensors for each sample using create_input_vector_wsi.py for the original WSIs and create_input_vector_patch.py for the WSIs divided into patches. They create a folder called wsi_tensors_100x and patches_tensors_100x respectively. Inside them all the sample input tensors in torch format are stored in the file format .pth and the file name is the corresponding sample name. The .pth file stores a list of two elements: a tensor matrix which contains the pixel values of each RGB channel (x) and a label (y). For this reason, samples are not divided into class folders.

2. Feature extraction and image classification.

Depending on the model, dataset and MIL embedding approach desired, you can use any of the files: tiny_vgg_patch_max.py, tiny_vgg_patch_mean.py, tiny_vgg_wsi, vgg_patch_max.py, vgg_patch_mean.py and vgg_wsi.py. To run them, you need to use a linux terminal and pass the root folder of the tensors dataset through the "-r" argument, e.g.:
python3 tiny_vgg_patch_max.py -r dataset/patches_tensors_100x
The script takes sample tensors as an input and prints a .pth file with the name of the model and stores several results in a dictionary. This dictionary contains several lists that stores the performance metrics results for each of the 100 epochs. The performance metrics are: loss, accuracy, precision, recall and specificity. It also stores the model with the updated weights.
