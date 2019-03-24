# MyoVision
# Description:
#   MyoVision is an image analysis tool designed to expedite muscle cross-section study

# How to Use:
#

# Table of Contents:
#   Use Ctrl+f to do a quick search for the following keywords. Include "START " at the beginning of the search.
#  (don't include quotes):
#       "imports"
#           Contains the import statements to obtain necessary functions of NumPy and SciPy
#       "functions"
#           Contains functions created for this project
#           "main"
#               Central driver function that runs when the program first runs. Calls other essential functions
#           "preprocess"
#               Automatically analyses image for preprocessing through image resizing based on current size
#
#
#


# IMPORTANT TO CITE
# https://scikit-learn.org/stable/about.html#citing-scikit-learn

# https://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting
#   The data is always a 2D array, shape (n_samples, n_features)
# https://docs.python.org/3/tutorial/classes.html



# START imports
from functions.display_mult_ims import display_mult_ims
from functions.lists2txt import lists2txt
from functions.pickle_helpers import save_object, load_object
from functions.im_overlay import im_overlay
from functions.sklearn_helpers import load_fiber_detect_training_set_from_array
from functions.image_processing import numpy_binary_to_pil
from PIL import Image, ImageDraw, ImageColor, ImageFont
from sklearn import svm
import math
import matplotlib.patches
import matplotlib.pyplot as plt

from MuscleStain import *

import multiprocessing as mp

import scipy as sp
import scipy.signal
import scipy.cluster
import skimage.color
import skimage.draw.draw
import skimage.exposure
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
import copy
import time
import os
# END imports

import numpy as np
import bisect
# from numba import jit


# START functions
# START construct_training_set*************************************************
# Description:
#     Given a file of blob metrics and a file of manually classified data, this
#     function parses the files and turns them into a matrix that has the
#     compiled metrics and manual classifications together.
# Inputs:
#     filename_class: text file of manually classified blobs by index.
#         WARNING, first line will be treated as a header line and will be ignored
#         Column 1 should contain blob indices of muscle fibers
#         Column 2 should contain blob indices of connected muscle fibers
#         Column 3 should contain blob indices of interstitial space
#         Column n... reserved for future use
#     filename_metrics: text file of blob metrics
#         Each column should consist of data for all blobs of a single classification
# Returns:
#     final: a matrix consisting of all metrics of filename_class as well as
#         all manually taken classifications of blobs
#         shape = [# classified blobs, (# metrics + 1)]
def construct_training_set(filename_class, filename_metrics):
    fiber_c = 1
    connected_c = 2
    space = 3

    compiled = sp.genfromtxt(filename_metrics, skip_header=1)
    num_metrics = compiled.shape[1]  # Number of columns

    # Open the classification file
    classification = open(filename_class, 'r')
    head_line = classification.readline()
    headers = head_line.split("\t")
    targets = []
    for line in classification:
        # Strips newline character from line
        line = line[:-1]
        # Splits line using tab delimiter
        metrics = line.split("\t")
        # Compile targets list, leaving a -1 to tell if there is nothing left in that classification
        temp_targets = []
        for entry in metrics:
            if entry is not '':
                temp_targets.append(int(entry))
            else:
                temp_targets.append(-1)
        # Add line to target matrix
        targets.append(temp_targets)

    targets = sp.array(targets)

    col = sp.empty([compiled.shape[0], 1])
    compiled = sp.concatenate((compiled, col), 1)

    for blob_num in targets[:, 0]:
        compiled[blob_num, num_metrics] = fiber_c

    for blob_num in targets[:, 1]:
        compiled[blob_num, num_metrics] = connected_c

    for blob_num in targets[:, 2]:
        compiled[blob_num, num_metrics] = space

    target_idxs = targets[targets >= 0]
    x = sp.arange(compiled.shape[0])
    x = sp.delete(x, target_idxs)
    final = sp.delete(compiled, x, axis=0)

    return final
# END construct_training_set///////////////////////////////////////////////////


# START construct_training_set_excel*******************************************
def construct_training_set_excel(classification_file, metric_file):
    fib_match_str = 'Muscle Fiber'
    connected_match_str = 'Merged'
    space_match_str = 'Interstitial Space '

    fib_target_num = 1
    connected_target_num = 2
    space_target_num = 3

    def clean_classifications(classifications):
        def parse_blob_num(str):
            num = str
            try:
                num = str.split("_")
                num = num[1]
            except TypeError:
                print("\'", num, "\' not parsed")
            return num

        # Get list of all good blob numbers from manual classification
        # good_blob_nums will be x-by-2 numpy array containing rows of (blob num, classification num)
        good_blob_nums = []
        # Go row by row
        for idx, series in classifications.iterrows():
            temp_row = []
            blob_label = series.iloc[1]
            blob_class = series.iloc[2]
            if blob_label is not None and blob_class is not None:
                blob_num = parse_blob_num(blob_label) # get blob number from this column
                try:
                    temp_row.append(int(blob_num))
                    # If classifications says 'Muscle Fiber'
                    if blob_class in fib_match_str:
                        # Check if it is also merged (classified as connected)
                        if series.iloc[3] is not None:
                            if connected_match_str in series.iloc[3]:
                                temp_row.append(connected_target_num)
                            else:
                                temp_row.append(fib_target_num)
                        # Otherwise classified as fiber
                        else:
                            temp_row.append(fib_target_num)
                    # If classifications says 'Interstitial Space', classify as space
                    elif blob_class in space_match_str:
                        temp_row.append(space_target_num)
                    # If no valid classification found, skip the fiber
                    else:
                        temp_row = None
                    # print(blob_num, "\t", series.iloc[1], "\t", series.iloc[2])
                    if temp_row is not None:
                        good_blob_nums.append(temp_row)
                except ValueError:
                    print("VALUE OF: ", blob_num, " SKIPPED.")

        good_blob_nums = sp.asarray(good_blob_nums)
        return good_blob_nums

    def clean_metrics(metrics):
        cols = list(metrics.iloc[1])
        metrics.columns = cols
        metrics = metrics.drop([0, 1])
        # metrics = metrics.to_numpy()
        return metrics

    def delete_repeats(classifications):
        to_del = []
        for b in range(len(classifications[:, 0])):
            for q in range(b+1, len(classifications[:, 0])):
                if classifications[b, 0] == classifications[q, 0]:
                    to_del.append(q)

        classifications = sp.delete(classifications, to_del, axis=0)
        return classifications

    wb_classification = openpyxl.load_workbook(classification_file)
    ws_classification = wb_classification[wb_classification.sheetnames[0]]
    classification_data = pd.DataFrame(ws_classification.values)
    classification_data = clean_classifications(classification_data) # list of blob nums and classifications
    classification_data = delete_repeats(classification_data)
    classification_data = classification_data[classification_data[:, 0].argsort()]
    cols = ['Blob Number', 'Target']

    wb_metrics = openpyxl.load_workbook(metric_file)
    ws_metrics = wb_metrics[wb_metrics.sheetnames[0]]
    metric_data = pd.DataFrame(ws_metrics.values)
    metric_data = clean_metrics(metric_data)
    cols.extend(list(metric_data.columns))
    metric_data = metric_data.to_numpy()

    idcs = []
    for blob_num in metric_data[:, 0]:
        if blob_num not in classification_data[:, 0]:
            idx_to_del = sp.where(metric_data[:, 0] == blob_num)
            idcs.append(idx_to_del)

    metric_data = sp.delete(metric_data, idcs, axis=0)
    # metric_data = sp.delete(metric_data, 0, axis=1)

    compiled = sp.concatenate((classification_data, metric_data), axis=1)
    compiled = pd.DataFrame(data=compiled)
    compiled.columns = cols

    return compiled
# END construct_training_set_excel/////////////////////////////////////////////


# START create_muscle_stains***************************************************
# Description:
#     Given a path to a folder of membrane stains, this function takes the images
#     in the folder and creates MuscleStain objects out of them before returning
#     a list of the stains
# Inputs:
#     folder_directory: path to folder containing membrane stain PNGs
# Returns:
#     stains: list of MuscleStain objects
def create_muscle_stains(folder_directory):
    st_idx = 0
    stains = []
    for im_name in os.listdir(folder_directory):
        if '.png' in im_name:
            im = plt.imread(folder_directory + "\\" + im_name)
            stains.append(MuscleStain(im))
            stains[st_idx].name = im_name.replace('.png','')
            st_idx = st_idx + 1

    return stains
# END create_muscle_stains/////////////////////////////////////////////////////


# START load_muscle_stains*****************************************************
def load_muscle_stains(folder_directory):
    st_idx = 0
    stains = []
    for file_name in os.listdir(folder_directory):
        if '.pkl' in file_name:
            stain = load_object(str(folder_directory + file_name))
            print('Loaded stain: ', stain.name)
            stains.append(stain)

    return stains
# END load_muscle_stains///////////////////////////////////////////////////////


# START save_stains************************************************************
# Description:
#     Pickles and saves all the MuscleStains in a list of MuscleStains
# Inputs:
#     stain_list: list of MuscleStains
# Returns:
#     N/A
def save_stains(stain_list):
    for stain in stain_list:
        print("Saving stain: ", stain.name)
        save_object(stain, stain.name)
# END save_stains//////////////////////////////////////////////////////////////


# START generate_stain_overlays************************************************
# Description:
#     Generates and displays/saves a blob overlay on original membrane image for
#     every MuscleStain in the list
# Inputs:
#     stains: a list of MuscleStain objects
#     display: When true, will display the end overlay for each stain individually
#     save: When true, will save the overlays as PNGs
# Returns:
#     N/A
def generate_stain_overlays(stains, display=False, save=True):
    for stain in stains:
        pil_im = numpy_binary_to_pil(stain.blob_labels)
        pil_im = stain.overlay_blobs(pil_im)
        ov = im_overlay(stain.membrane_image, pil_im)

        if display:
            display_mult_ims(ov)
        if save:
            ov.save(stain.name + "_overlay.png", "PNG")


# END generate_stain_overlays//////////////////////////////////////////////////


def parallel_test(*stains):
    def ff(stain):
        stain.find_fibers()

    if __name__ == '__main__':
        pool = mp.Pool()
        pool.map(ff, stains)



# START stain_operate**********************************************************
def stain_operate(stains, process_name, save=None, **kwargs):
    start_s = time.time()
    for stain in stains:
        stain.process(
            kwargs
        )
# END stain_operate////////////////////////////////////////////////////////////

# START fib_excel_setup********************************************************
def fib_excel_setup(stains, excel_file):
    start_s = time.time()
    for stain in stains:
        stain.initialize_excel_data(filename=excel_file)
    end_s = time.time()
    print("Excel setup of ", len(stains), " stains: ", end_s-start_s, " seconds.")
# END fib_excel_setup//////////////////////////////////////////////////////////

# START fib_detect*************************************************************
# Description:
#     Runs the find_fibers() member function on a number of MuscleStain objects
# Inputs:
#     stains: a list of MuscleStain objects
#     save: When true, pickles the MuscleStains and saves them as a .pkl files
#         using the fiber's name as the filename
# Returns:
#     N/A
def fib_detect(stains, save=None):
    start_s = time.time()
    for stain in stains:
        stain.find_fibers()
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
    end_s = time.time()
    print("Fiber detection of ", len(stains), " stains: ", end_s-start_s, " seconds.")
# END fib_detect///////////////////////////////////////////////////////////////


# START fib_info*************************************************************
# Description:
#     Runs the find_fiber_properties() member function on a number of MuscleStain
#     objects
# Inputs:
#     stains: a list of MuscleStain objects
#     save: When true, pickles the MuscleStains and saves them as a .pkl files
#         using the fiber's name as the filename
# Returns:
#     N/A
def fib_info(stains, rerun_connected_fibs=False, save=None):
    start_s = time.time()
    for stain in stains:
        stain.find_fiber_properties(rerun_connected=rerun_connected_fibs)
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
    end_s = time.time()
    print("Fiber properties of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_info/////////////////////////////////////////////////////////////////


# START fib_classify***********************************************************
def fib_classify(stains, classifier, save=None):
    start_s = time.time()
    for stain in stains:
        stain.classify_blobs(classifier)
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
    end_s = time.time()
    print("Fiber classifying of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_classify/////////////////////////////////////////////////////////////


# START fib_masks**************************************************************
def fib_masks(stains, save=None):
    start_s = time.time()
    for stain in stains:
        stain.get_masks()
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
    end_s = time.time()
    print("Fiber masking of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_masks////////////////////////////////////////////////////////////////


# START fib_separate***********************************************************
def fib_separate(stains, h_t, save=None):
    start_s = time.time()
    for stain in stains:
        stain.separate_connected_fibers(h_test=h_t)
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
        end_s = time.time()
    print("Watershed segmenting of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_separate/////////////////////////////////////////////////////////////


# START fib_separate***********************************************************
def fib_refine(stains, save=None):
    start_s = time.time()
    for stain in stains:
        stain.refine_contours()
        if save:
            print("Saving stain: ", stain.name)
            save_object(stain, stain.name, save)
        end_s = time.time()
    print("Active Contour of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_separate/////////////////////////////////////////////////////////////


# START fib_write_data*********************************************************
def fib_write_data(stains):
    start_s = time.time()
    for stain in stains:
        stain.write_data()
    end_s = time.time()
    print("Data write of ", len(stains), " stains: ", end_s - start_s, " seconds.")
# END fib_write_data///////////////////////////////////////////////////////////


# START obtain_stain_metrics***************************************************
def obtain_stain_metrics(folder_from, save_to=None):
    stains = []
    for im_name in os.listdir(folder_from):
        if '.png' in im_name:
            im = plt.imread(folder_from + "\\" + im_name)
            # im = im[1000:3000, 1000:3000, :]
            musc = MuscleStain(im)
            musc.name = im_name.replace('.png', '')
            if save_to:
                folder_to_save = save_to
            else:
                folder_to_save = ""
            stains = [musc]

            fib_excel_setup(stains, excel_file=("" + folder_to_save + musc.name))
            fib_detect(stains)
            fib_info(stains, rerun_connected_fibs=False)
            fib_write_data(stains)
# END obtain_stain_metrics/////////////////////////////////////////////////////


# START run_folder*************************************************************
def run_folder(folder_from, clf, save_to=None):
    stains = []
    for im_name in os.listdir(folder_from):
        if '.png' in im_name:
            start_s = time.time()

            im = plt.imread(folder_from + "\\" + im_name)
            im = im[1000:3000, 1000:3000, :]
            musc = MuscleStain(im)
            musc.name = im_name.replace('.png', '')
            if save_to:
                folder_to_save = save_to
            else:
                folder_to_save = ""
            stains = [musc]

            fib_excel_setup(stains, excel_file=("" + folder_to_save + musc.name))
            fib_detect(stains)
            fib_info(stains, rerun_connected_fibs=False)
            fib_classify(stains, clf)
            fib_masks(stains)
            fib_separate(stains, 1)
            fib_info(stains, rerun_connected_fibs=True)
            fib_classify(stains, clf)
            fib_masks(stains)
            fib_refine(stains, save=folder_to_save)
            # stains[0].display_contours()
            fib_write_data(stains)

            # save_object(musc, str(musc.name + "_prewatershed"))
            # for h in sp.arange(0, 5, 0.5):
            #     print("Testing h of: ", h)
            #     temp_musc = copy.deepcopy(musc)
            #     temp_stains = [temp_musc]
            #     fib_separate(temp_stains, h)
            #     musc_name = "h_of_" + str(h) + "_" + temp_musc.name
            #     print(musc_name)
            #     fib_info(temp_stains, rerun_connected_fibs=True, dump_m=True, dump_im=True, file_names=(""+folder_to_save+musc_name))
            #     fib_classify(temp_stains, clf)
            #     fib_masks(temp_stains)
            #     save_object(temp_musc, "D:\\Documents\\PythonCode\\MyoVision\\watershed_testing\\after_improc\\" + musc_name)

            end_s = time.time()
            print("Full processing of ", musc.name, " completed in:\n\t", end_s-start_s, " seconds.")
# END run_folder///////////////////////////////////////////////////////////////


# START main*******************************************************************
# def main():
    # Variables
# membrane_file_name = "D:\\Documents\\UK\\Semester3\\EGR390\\MyoVisionIms\\UABMAS 334-1 20xMosiaX CMD-Image Export-04_FITC.png"
# membrane_file_name2 = "D:\\Documents\\UK\\Semester3\\EGR390\\MyoVisionIms\\PoWeR 2 Gastroc 10x FT-Specific Myonuclei-Image Export-02_Texas Red.png"
# membrane_image = plt.imread(membrane_file_name)
# membrane_image2 = plt.imread(membrane_file_name2)
# print(membrane_image.shape)
# membrane_image = membrane_image[1000:4000, 1000:4000, :]
# membrane_image2 = membrane_image2[1000:4000, 1000:4000, :]
#
# s = MuscleStain(membrane_image)
# a = MuscleStain(membrane_image2)
#
# stains = [s, a]
# parallel_test(stains)

# serial_test(s, a)

def ff(stain):
    print("hey")
    stain.find_fibers()

# START main********************************************************************
if __name__ == '__main__':

    print("not broke yet")
    # obtain_stain_metrics("Processed\\membrane_ims", save_to='metric_results\\')
    # exit(0)


    all_d1 = construct_training_set_excel(classification_file="raw_manual_classifications\\(DATA)UABMAS334 Biopsy 1_10xMosiaX CM.xlsx",
                                         metric_file="metric_results\\UABMAS334 Biopsy 1_10xMosiaX CMD-Image Export-06_DAPI.xlsx")
    print(all_d1.columns)
    d1, t1 = load_fiber_detect_training_set_from_array(all_d1, target_col=1, ignore_cols=[0, 2])

    print("Constructing training set")
    all_d2 = construct_training_set_excel(classification_file="raw_manual_classifications\\PoWeR 2 Gastroc 10x FT-Specific Myonuclei-Image Export-02_Texas Red(MINE).xlsx",
                           metric_file="metric_results\\PoWeR 2 Gastroc 10x FT-Specific Myonuclei-Image Export-02_Texas Red.xlsx")
    d2, t2 = load_fiber_detect_training_set_from_array(all_d2, target_col=1, ignore_cols=[0, 2])

    d = sp.concatenate((d1, d2), axis=0)
    t = sp.concatenate((t1, t2), axis=0)
    # print(d.shape)
    # print(t.shape)
    # d, t = load_fiber_detect_training_set_from_array(all_d, 5, ignore_cols=[0,1])

    d = d.tolist()
    t = t.tolist()

    clf = svm.SVC(gamma=0.001, C=100.)
    # print(all_d)
    # print(d)
    # print(t)
    clf.fit(d[:], t[:])

    # loaded = load_object('D:\Documents\PythonCode\MyoVision\precontour_test\PoWeR 3 - Gastroc 10x Fiber-Type-Image Export-03_DAPI.pkl')
    # loaded = load_object('D:\Documents\PythonCode\MyoVision\save_test\PoWeR 3 - Gastroc 10x Fiber-Type-Image Export-03_DAPI.pkl')
    # loaded = load_object('D:\Documents\PythonCode\MyoVision\Processed_Feb_10_2019\PoWeR 3 - Gastroc 10x Fiber-Type-Image Export-03_DAPI.pkl')
    # display_mult_ims(loaded.membrane_image)
    # loaded.display_contours()
    # print(type(loaded))
    # test = [loaded]
    # fib_masks(test)
    # test[0].display_classifications()
    # loaded.refine_contours()
    # loaded.display_contours()
    # loaded.interactive_blobs()
    # display_mult_ims(loaded.membrane_image)
    # loaded.display_contours()

    run_folder("testing", clf, save_to='test_results\\')
# END main/////////////////////////////////////////////////////////////////////
# END functions
