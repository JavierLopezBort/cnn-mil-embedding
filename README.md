# cnn-mil-embedding
Classification of Whole Slide Images from HNSCC using CNN-MIL embedding-based predicting models

There are two main parts in this repository: dataset creation and feature extraction / image classification

1. Dataset creation

All the necessary scripts for this part are inside the "dataset" folder. You can follow these steps:

1.1 Create a root folder where you can store all the raw WSI. In our case is called "wsi_100x". Here, you should create two subfolders with the
corresponding name of the classes and store the raw images inside them.

1.2 Now you can oversample if necessary. In this case, the "data_augmentation.py" file creates the new WSI images and store them in the
same root folder. You can decide the number of times that you want to oversample with the variable "increase_size".

1.3 Now you can downsample if necessary. In this case, the "downsampling.py" selectes randomly a specific number of WSI from the majority class and
deletes them to have the exact number of cases in both classes. Again, this is done in the original root folder.

1.4 To apply a MIL model, you need first to divide the WSI into patches. This is done by the "openslide.py" file, which creates a folder called
"patches_100x" where it stores all the patches from a WSI into a folder with the sample name. It has the same folder structure as "wsi_100x".

1.5 
