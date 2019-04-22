# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:46 2019

@author: kscamp3
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_processing.image_proc as im_proc
import machine_learning.machine_learn as ml

if __name__ == "__main__":
    
    if (0):
        excel_file_string = '..\\train\\train_02.xlsx'
        output_classifier_file_string = '..\\temp\\classifier_2.svc'
        
        ml.learn_test_1(excel_file_string, output_classifier_file_string)
    
    if (0):
        im_proc.shuffle_labeled_image(np.ones(3))
    
    if (1):
        # implement classifier
        im_file_string = '..\\data\\tiny_test_image_flipped.png'
        classifier_file_string = '..\\temp\\classifier_2.svc'
        blob_excel_file_string = '..\\temp\\excel_02.xlsx'
        
        im_mask, im_label, blob_data, region = \
            ml.implement_classifier(im_file_string, classifier_file_string)
            
        im_proc.handle_potentially_connected_fibers(im_mask,
                                                    im_label,
                                                    blob_data,
                                                    region)
        
    #    im_label = im_proc.raw_image_to_labeled_image(im_file_string)
    #    
    #    im_proc.calculate_blob_properties(im_label,
    #                                      output_excel_file_string = blob_excel_file_string)
        
    
    if (0):
        
        # Variables
        im_file_string = '..\\data\\tiny_test_image.png'
        saturation_percent = 5
        min_blob_area = 5
        image_base_file_string = "..\\temp\\ken_test_image_3"
        excel_file_string = "..\\train\\train_03.xlsx"
        
        im_label, im_sat = im_proc.raw_image_file_to_labeled_image(im_file_string)
        
        im_proc.calculate_blob_properties(im_label,
                                          output_image_base_file_string = image_base_file_string,
                                          im_gray = im_sat,
                                          output_excel_file_string = excel_file_string)
