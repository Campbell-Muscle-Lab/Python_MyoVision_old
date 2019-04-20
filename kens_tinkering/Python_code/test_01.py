# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:46 2019

@author: kscamp3
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_processing.image_processing as im_proc

# Variables
im_file_string = '..\\data\\tiny_test_image.png'

# Read in image
im = cv2.imread(im_file_string)

# Convert to gray-scale and normalize
im_gray = im_proc.return_gray_scale_image(im)
im_norm = im_proc.normalize_gray_scale_image(im_gray)
im_sat = im_proc.saturate_gray_scale_image(im_norm, 5)

# Return edge
im_frangi = im_proc.apply_Frangi_filter(im_sat)
im_edge = im_proc.otsu_threshold(im_frangi)

im_clear_edge = im_proc.clear_edge(im_edge, invert_mode=0)

im_label = im_proc.label_image(im_clear_edge)

im_rgb_label = im_proc.label_to_rgb(im_label)

r= im_proc.deduce_region_props(im_label)

im_proc.show_blobs(im_label,r)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13,3), ncols=3)
ax1.imshow(im)

pos = ax2.imshow(im_gray)
fig.colorbar(pos,ax = ax2)

pos = ax3.imshow(im_norm)
fig.colorbar(pos, ax = ax3)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

ax1.imshow(im_sat)
fig.colorbar(pos, ax=ax1)

ax2.imshow(im_frangi)
fig.colorbar(pos, ax=ax2)

ax3.imshow(im_edge)
fig.colorbar(pos, ax=ax3)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
ax1.imshow(im_clear_edge)

ax2.imshow(im_label)

ax3.imshow(im_rgb_label)