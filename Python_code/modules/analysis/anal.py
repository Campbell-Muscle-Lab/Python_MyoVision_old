# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:21:07 2019

@author: kscamp3
"""

from modules.image_processing import image_proc as im_proc
from modules.machine_learning import machine_learn as ml
from modules.untangle import untangle as ut
from modules.xml import xml as x

import os, shutil
import numpy as np

def find_blobs(configuration_file):
    # Find individual blobs in an image file

    # Parse the analysis xml file into a doc
    doc = ut.parse(configuration_file)

    # Deduce the input image file and the results folder
    task_list = x.unpack_task_files_xml(doc)

    # Unpack the xml into dictionaries
    image_to_label_parameters = x.unpack_image_to_label_parameters_xml(doc)

    # Loop through blobs
    for i, t in enumerate(task_list):
        print('Finding blobs in  %s with results written to %s' %
              (task_list[i].raw_image_file_string,
               task_list[i].results_folder))

        # Pull out results_folder for later use
        results_folder = task_list[i].results_folder

        # Create a labeled image
        im_label, im_sat, im_gray = im_proc.raw_image_file_to_labeled_image(
                        task_list[i].raw_image_file_string,
                        image_to_label_parameters=image_to_label_parameters,
                        results_folder=results_folder)

        # Process the image, saving individual blobs as we go
        output_excel_file_string = os.path.join(results_folder,
                                                'blob_metrics.xlsx')
        output_blob_base_file_string = os.path.join(results_folder,
                                                    'blobs','blob')

        blob_data, regions = im_proc.calculate_blob_properties(
                im_label,
                output_excel_file_string=output_excel_file_string,
                output_blob_base_file_string=output_blob_base_file_string,
                im_base=im_gray)

        # Overlay the labeled image on the background
        im_temp = im_proc.shuffle_labeled_image(im_label,
                                                bg_color=(0,0,0))
        im_b = im_proc.merge_rgb_planes(np.zeros(im_gray.shape),
                                        np.zeros(im_gray.shape),
                                        im_proc.normalize_gray_scale_image(im_gray))
        im_overlay = np.ubyte(0.5 * 255 * im_temp + 0.5 * 255 * im_b)


        annotated_image_file_string = os.path.join(results_folder,
                                                   'annotated_overlay.png')
        
        im_proc.create_annotated_blob_overlay(im_overlay, regions,
                                              annotated_image_file_string)


def train_classifier(configuration_file):
    # Train the classifier

    # Parse the analysis xml file into a doc
    doc = ut.parse(configuration_file)

    # Unpack xml doc into dictionaries
    train_classifier_parameters = x.unpack_train_classifier_parameters_xml(doc)

    # Train classifier
    ml.create_classifier_model(train_classifier_parameters)


def analyze_images(configuration_file):
    # Parses the task list, and then analyzes images in series
    
    # Parse the configuration_file into dictionaries
    doc = ut.parse(configuration_file)

    # Deduce the input image file and the results folder
    task_list = x.unpack_task_files_xml(doc)

    for i, t in enumerate(task_list):
        print('Analyzing %s with results written to %s' %
              (task_list[i].raw_image_file_string,
               task_list[i].results_folder))

        analyze_image_file(task_list[i].raw_image_file_string,
                           task_list[i].results_folder,
                           configuration_file)

def analyze_image_file(raw_image_file_string,
                       results_folder,
                       configuration_file):
    # Analyzes an image file, writes results to folder, using the
    # parameters specified in the configuration file

    # Parse the configuration_file into dictionaries
    doc = ut.parse(configuration_file)
    # Unpack parameters into dictionaries
    image_to_label_parameters = x.unpack_image_to_label_parameters_xml(doc)
    classifier_parameters = x.unpack_classifier_parameters_xml(doc)
    refine_fibers_parameters = x.unpack_refine_fibers_parameters_xml(doc)

    # Process image file
    im_final_classification, im_final_label, im_final_overlay = \
        ml.implement_classifier(raw_image_file_string,
                                results_folder=results_folder,
                                image_to_label_parameters=image_to_label_parameters,
                                classifier_parameters=classifier_parameters,
                                refine_fibers_parameters=refine_fibers_parameters)

    # Write fiber data to file
    output_excel_file_string = os.path.join(results_folder,
                                            'final_results.xlsx')
    [blob_data, regions] = im_proc.calculate_blob_properties(im_final_label,
                                 output_excel_file_string=output_excel_file_string)

    # Create the final overlay images
    overlay_image_file_string = os.path.join(results_folder,
                                             'clean_overlay.png')
    im_proc.write_image_to_file(im_final_overlay, overlay_image_file_string)

    annotated_image_file_string = os.path.join(results_folder,
                                               'annotated_overlay.png')
    im_proc.create_annotated_blob_overlay(im_final_overlay, regions,
                                          annotated_image_file_string)

    # Zip the processing folder
    print('Zipping processing folder')
    make_archive(os.path.join(results_folder, 'processing'),
                 (os.path.join(results_folder,'processing.zip')))
    # Delete the processing folder
    print('Removing raw processing files')
    shutil.rmtree(os.path.join(results_folder, 'processing'))

    # Tidy up
    print('__')
    print('%s analyzed, results written to %s' %
          (raw_image_file_string, results_folder))

def make_archive(source, destination):
    # Code from here
    # http://www.seanbehan.com/how-to-use-python-shutil-make_archive-to-zip-up-a-directory-recursively-including-the-root-folder/
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    print(source, destination, archive_from, archive_to)
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)
