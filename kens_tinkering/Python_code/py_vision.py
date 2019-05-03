# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:46 2019

@author: kscamp3
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

import modules.image_processing.image_proc as im_proc
import modules.machine_learning.machine_learn as ml
import modules.analysis.anal as an

if __name__ == "__main__":
    
    if (0):
        excel_file_string = '..\\train\\Power_3_Gastroc_10x_blue_cropped_assigned.xlsx'
        output_classifier_file_string = '..\\classifier\\Power_3_Gastroc_10x_blue_cropped_linear.svc'
        
        ml.learn_test_2(excel_file_string, output_classifier_file_string)
    
    if (0):
        im_file_string = '..\\data\\top_left_Power_3_Gastroc_10x_blue.png'
        im_proc.kens_test(im_file_string)
    
    if (1):
        # implement classifier
        im_file_string = '..\\data\\Power_3_Gastroc_10x_blue_cropped.png'
#        im_file_string = '..\\data\\Power_3_Gastroc_10x_blue_cropped_small.png'

        image_to_label_parameters={}
        image_to_label_parameters['saturation_percent'] = 15
        image_to_label_parameters['min_object_size'] = 50
        image_to_label_parameters['verbose_mode'] = 1
        image_to_label_parameters['troubleshoot_mode'] = 1
        image_to_label_parameters['block_size']=1000
        image_to_label_parameters['process_image_to_blobs_file_string'] = \
            "..\\temp\\blocks\\process_blocks"

        classifier_parameters={}
        classifier_parameters['classifier_file_string'] = \
            '..\\classifier\\Power_3_Gastroc_10x_blue_cropped_linear.svc'
        classifier_parameters['verbose_mode'] = 1
        classifier_parameters['watershed_distance'] = 10
        classifier_parameters['classification_steps_image_file_string'] = \
            '..\\temp\\classification\\classification_steps.png'

        refine_fibers_parameters = {}
        refine_fibers_parameters['max_iterations']= 25
        refine_fibers_parameters['lambda2']=2
        refine_fibers_parameters['refine_fibers_image_file_string'] = \
            '..\\temp\\refine_edges\\refine_fibers.png'

        results_parameters={}
        results_parameters['overlay_image_file_string'] = \
            final_overlay_file_string = '..\\temp\\final_overlay.png'

        an.analyze_image_file(im_file_string,
                              image_to_label_parameters=image_to_label_parameters,
                              classifier_parameters=classifier_parameters,
                              refine_fibers_parameters=refine_fibers_parameters,
                              results_parameters=results_parameters)


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
