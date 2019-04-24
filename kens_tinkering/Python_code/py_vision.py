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
        im_proc.kens_test()
    
    if (0):
        # implement classifier
        im_file_string = '..\\data\\tiny_test_image_flipped.png'
        classifier_file_string = '..\\temp\\classifier_2.svc'
        
        ml.implement_classifier(im_file_string, classifier_file_string)
#            
#        im_proc.handle_potentially_connected_fibers(im_mask,
#                                                    im_label,
#                                                    blob_data,
#                                                    region)
#        
#    #    im_label = im_proc.raw_image_to_labeled_image(im_file_string)
#    #    
#    #    im_proc.calculate_blob_properties(im_label,
#    #                                      output_excel_file_string = blob_excel_file_string)
#        
    
    if (1):
        raw_im_file_string = '..\\data\PoWer_3_Gastroc_10x_blue.png'
        saturation_percent = 15
        min_blob_area = 5
        label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled_cropped.png'
        shuffled_label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled_cropped_shuffled.png'
        
        im_label, im_sat, im_shuffled = \
            im_proc.raw_image_file_to_labeled_image(raw_im_file_string,
                                                    saturation_percent = saturation_percent,
                                                    min_object_size = min_blob_area,
                                                    verbose_mode = 1,
                                                    image_label_file_string = label_file_string,
                                                    image_shuffled_label_file_string = shuffled_label_file_string)
#        
#        im_proc.calculate_blob_properties(im_label,
#                                          output_image_base_file_string = image_base_file_string,
#                                          im_gray = im_sat,
#                                          output_excel_file_string = excel_file_string)


    
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
