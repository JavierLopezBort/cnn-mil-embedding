# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:07:59 2023

@author: jalob
"""

import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

sample_path = "dataset/wsi_100x/*/*"

sample_path_list = glob.glob(sample_path)

root = "patches_100x"

os.makedirs(root)

for sample_idx, sample in enumerate(sample_path_list):
    
    ##################################################
    # 1. PREPROCESSING
    ##################################################
    
    # Create the sample folder to store the patches
    sample_name = re.search(r"dataset/wsi_100x(.+)\.jpg", sample).group(1)
    
    sample_folder = root + sample_name
    
    os.makedirs(sample_folder)
    
    #Load the slide file (svs) into an object.
    slide = open_slide(str(sample))
    
    patch_size = 256
    #Generate object for tiles using the DeepZoomGenerator
    tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=False)
    #Here, we have divided our svs into tiles of size 256 with no overlap. 
    
    max_level = int(tiles.level_count) - 1
    
    # Take the number of rows and columns for the maximum level
    cols, rows = tiles.level_tiles[max_level]
    
    ##################################################
    # 2. PATCH EXTRACTION
    ##################################################
    
    patch_idx = 1
    
    for row in range(rows):
        for col in range(cols):
            
            # Create the tiles for each row and col
            patch = tiles.get_tile(max_level, (col,row))
            patch_RGB = patch.convert('RGB')
            patch_np = np.array(patch_RGB)
            
            plt.imsave(sample_folder + "/" + str(patch_idx) + ".png", patch_np)
            print("N:", sample_idx + 1, ", sample:", sample_name, ", patch:", str(patch_idx))
            
            patch_idx += 1