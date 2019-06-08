# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:21:07 2019

@author: kscamp3
"""

from modules.image_processing import image_proc as im_proc
from modules.machine_learning import machine_learn as ml
from modules.untangle import untangle as ut
from modules.xml import xml as x

def find_blobs(configuration_file):
    # Find individual blobs in an image file

    # Parse the analysis xml file into a doc
    doc = ut.parse(configuration_file)
    
    # Unpack xml doc into a filename and dictionaries
    raw_image_file_string = doc.MyoVision_analysis.raw_image.image_file_string.cdata
    image_to_label_parameters = x.unpack_image_to_label_parameters_xml(doc)
    calculate_blob_parameters = x.unpack_calculate_blob_parameters_xml(doc)

    # Create a labeled image
    im_label, im_sat, im_gray = im_proc.raw_image_file_to_labeled_image(
                        raw_image_file_string,
                        image_to_label_parameters=image_to_label_parameters)

    # Procss the image to get blobs
    blob_data, regions = im_proc.calculate_blob_properties(im_label, im_sat,
                                      calculate_blob_parameters=calculate_blob_parameters)

    # Make an annoted overlay
    im_proc.create_annotated_blob_overlay(im_label, im_sat, regions,
                                          calculate_blob_parameters)


def train_classifier(configuration_file):
    # Train the classifier

    # Parse the analysis xml file into a doc
    doc = ut.parse(configuration_file)

    # Unpack xml doc into dictionaries
    train_classifier_parameters = x.unpack_train_classifier_parameters_xml(doc)

    # Train classifier
    ml.create_classifier_model(train_classifier_parameters)


def analyze_image_file(configuration_file):
    # Analyzes an image file
    
    # Parse the configuration_file into dictionaries
    doc = ut.parse(configuration_file)

    raw_image_file_string = doc.MyoVision_analysis.raw_image.image_file_string.cdata
    image_to_label_parameters = x.unpack_image_to_label_parameters_xml(doc)
    classification_parameters = x.unpack_classification_parameters_xml(doc)
    refine_fibers_parameters = x.unpack_refine_fibers_parameters_xml(doc)
    results_parameters = x.unpack_results_parameters_xml(doc)

    # Process image file
    im_final_classification, im_final_label, im_final_overlay = \
        ml.implement_classifier(raw_image_file_string,
                            image_to_label_parameters=image_to_label_parameters,
                            classification_parameters=classification_parameters,
                            refine_fibers_parameters=refine_fibers_parameters)
    
    im_proc.write_image_to_file(im_final_overlay,
                                results_parameters['overlay_image_file_string'])
