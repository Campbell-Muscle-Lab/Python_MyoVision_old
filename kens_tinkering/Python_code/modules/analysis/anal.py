# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:21:07 2019

@author: kscamp3
"""

from modules.image_processing import image_proc as im_proc
from modules.machine_learning import machine_learn as ml


def analyze_image_file(im_file_string,
                       image_to_label_parameters=[],
                       classifier_parameters=[],
                       refine_fibers_parameters=[],
                       results_parameters=[]):
    # Top level file to analyze a single image
    
    im_final_classification, im_final_label, im_final_overlay = \
        ml.implement_classifier(im_file_string,
                            image_to_label_parameters=image_to_label_parameters,
                            classifier_parameters=classifier_parameters,
                            refine_fibers_parameters=refine_fibers_parameters)
    
    im_proc.write_image_to_file(im_final_overlay,
                                results_parameters['overlay_image_file_string'])
