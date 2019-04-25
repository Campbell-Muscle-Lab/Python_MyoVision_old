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
        excel_file_string = '..\\train\\Power_3_Gastroc_10x_blue_cropped_assigned.xlsx'
        output_classifier_file_string = '..\\classifier\\Power_3_Gastroc_10x_blue_cropped_assigned.svc'
        
        ml.learn_test_1(excel_file_string, output_classifier_file_string)
    
    if (0):
        im_proc.kens_test()
    
    if (1):
        # implement classifier
        im_file_string = '..\\data\\Power_3_Gastroc_10x_blue_cropped.png'
        classifier_file_string = '..\\classifier\\Power_3_Gastroc_10x_blue_cropped_assigned.svc'
        saturation_percent = 15
        min_blob_area = 50
        image_label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled.png'
        shuffled_label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled.png'
        
        classifier_parameters={}
        classifier_parameters['verbose_mode'] = 1
        classifier_parameters['watershed_distance'] = 10
        
        label_image_parameters={}
        label_image_parameters['saturation_percent'] = 15
        label_image_parameters['min_object_size'] = 50
        label_image_parameters['verbose_mode'] = 1
        label_image_parameters['troubleshoot_mode'] = 1
        label_image_parameters['image_label_file_string'] = \
            image_label_file_string
        label_image_parameters['shuffled_label_file_string'] = \
            shuffled_label_file_string

        ml.implement_classifier(im_file_string,
                                classifier_file_string,
                                classifier_parameters = classifier_parameters,
                                image_to_label_parameters=label_image_parameters)
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
    
    if (0):
        raw_im_file_string = '..\\data\PoWer_3_Gastroc_10x_blue_cropped.png'
        saturation_percent = 15
        min_blob_area = 50
        label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled_cropped.png'
        shuffled_label_file_string = '..\\temp\\Power_3_Gastroc_10x_blue_labeled_cropped_shuffled.png'
        
        image_base_file_string = '..\\temp\\temp2\\blob'
        excel_file_string = '..\\train\\PoWer_3_Gastroc_10x_blue_cropped.xlsx'
        
        im_label, im_sat, im_shuffled, im_gray = \
            im_proc.raw_image_file_to_labeled_image(raw_im_file_string,
                                                    saturation_percent = saturation_percent,
                                                    min_object_size = min_blob_area,
                                                    verbose_mode = 1,
                                                    image_label_file_string = label_file_string,
                                                    image_shuffled_label_file_string = shuffled_label_file_string)

        im_proc.calculate_blob_properties(im_label,
                                          output_image_base_file_string = image_base_file_string,
                                          im_base = im_gray,
                                          output_excel_file_string = excel_file_string)



    
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
