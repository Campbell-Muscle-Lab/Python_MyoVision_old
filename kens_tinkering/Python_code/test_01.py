# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:46 2019

@author: kscamp3
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_processing.image_processing as im_proc
import machine_learning.machine_learning as ml

if (0):
    excel_file_string = '..\\train\\train_01.xlsx'
    output_classifier_file_string = '..\\temp\\classifier.svc'
    
    ml.learn_test_1(excel_file_string, output_classifier_file_string)
    
if (1):
    # implement classifier
    im_file_string = '..\\data\\tiny_test_image_flipped.png'
    blob_excel_file_string = '..\\temp\\excel_flipped.xlsx'
    
    a = ml.implement_classifier(im_file_string)
    
#    im_label = im_proc.raw_image_to_labeled_image(im_file_string)
#    
#    im_proc.calculate_blob_properties(im_label,
#                                      output_excel_file_string = blob_excel_file_string)
    

if (0):
    
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
    
    im_remove_small_objects = im_proc.remove_small_objects(im_clear_edge, 5)
    
    im_label = im_proc.label_image(im_remove_small_objects)
    
    im_rgb_label = im_proc.label_to_rgb(im_label)
    
    r= im_proc.deduce_region_props(im_label)
    
    im_proc.calculate_blob_properties(im_sat,im_label,
                       output_image_base_file_string="..\\temp\\ken",
                       output_excel_file_string = "..\\temp\\data.xlsx")

#fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13,3), ncols=3)
#ax1.imshow(im)
#
#pos = ax2.imshow(im_gray)
#fig.colorbar(pos,ax = ax2)
#
#pos = ax3.imshow(im_norm)
#fig.colorbar(pos, ax = ax3)
#
#fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
#
#ax1.imshow(im_sat)
#fig.colorbar(pos, ax=ax1)
#
#ax2.imshow(im_frangi)
#fig.colorbar(pos, ax=ax2)
#
#ax3.imshow(im_edge)
#fig.colorbar(pos, ax=ax3)
#
#fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
#ax1.imshow(im_clear_edge)
#
#ax2.imshow(im_label)
#
#ax3.imshow(im_rgb_label)