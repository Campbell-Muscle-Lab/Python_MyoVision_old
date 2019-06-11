# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:53:42 2019

@author: kscamp3
"""
import scipy as sp
import numpy as np
from skimage.color import rgb2gray
from skimage.util import invert
import os
import matplotlib.pyplot as plt

from modules.machine_learning import machine_learn as ml


def write_image_to_file(im, im_file_string, verbose=1):
    # Writes an image to file

    from skimage.io import imsave

    # Check directory exists and save image file
    dir_path = os.path.dirname(im_file_string)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if (verbose):
        print('Writing image to %s' % im_file_string)

    imsave(im_file_string, im)


def return_gray_scale_image_as_float(rgb_image):
    # converts a color image to a gray-scale float image

    from skimage import img_as_float

    return img_as_float(rgb2gray(rgb_image))


def normalize_gray_scale_image(gray_image):
    # normalizes a gray-scale image between 0 and 1
    norm_image = gray_image - np.amin(gray_image)
    norm_image = norm_image / np.amax(norm_image)
    return norm_image


def saturate_gray_scale_image(gray_image, x):
    # saturates a gray-scale image so that x% of pixels are at low or high
    from skimage.exposure import rescale_intensity

    # This code starts by attempting to rescale at x%
    # If that's not possible (for example, the image is mostly dark, with
    # a small bright area), lo,hi reqturn as 0 and sat_im becomes NaN throwing
    # an error
    # To overcome this, we start with an initial value of x, and check the
    # saturation works. If it doesn't, we reduce x successively. If x drops
    # too low, sat_im is set all zeros

    initial_x = x

    keep_going = 1
    while (keep_going):
        lo, hi = np.percentile(gray_image, (x, 100-x))

        if (hi > 0):
            keep_going = 0
            sat_im = rescale_intensity(gray_image, in_range=(lo, hi))
        else:
            x = x - (0.1*initial_x)
            # Final check
            if (x <= 0):
                keep_going = 0
                print('saturate_gray_scale_image was forced to return zeros')
                sat_im = np.zeros(gray_image.shape)

    return sat_im


def apply_Frangi_filter(im,
                        scale_low=1, scale_high=10, scale_step=1):
    # applies a Frangi filter to detect edges
    from skimage.filters import frangi

    # This step seems to be important,
    # Frangi doesn't work well with (0, 1) image
    im = 255.0 * im

    # Lowering scale range below 1 seems to produce noise
    # Upper scale range gives wider boundaries between cells
    im_out = frangi(invert(im),
                    np.arange(scale_low, scale_high, scale_step))

    return im_out


def otsu_threshold(gray_image):
    # implements an otsu threshold
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(gray_image)
    im_binary = gray_image > thresh
    im_binary = im_binary.astype(sp.bool_)
    im_binary = invert(im_binary)

    return im_binary


def label_image(im, connectivity=1):
    from skimage.measure import label

    im_label = label(im, connectivity=1)
    return im_label


def label_to_rgb(im_label):
    # turns a labeled image into an rgb image

    from skimage.color import label2rgb

    return label2rgb(im_label)


def clear_edge(im, invert_mode=0):
    # clears edge
    from skimage.segmentation import clear_border

    if (invert_mode):
        im = invert(im)

    return clear_border(im)


def k_means_image(im, no_of_clusters):
    # separates a gray-scale image into no_of_clusters values
    from sklearn.cluster import KMeans

    z = im.reshape((-1, 1))
    k_means = KMeans(n_clusters=no_of_clusters, random_state=0).fit(z)
    print(k_means)
    im_out = k_means.labels_.reshape((im.shape))
    return im_out


def remove_small_objects(im, size_threshold):
    # removes objects below size_threshold
    from skimage.morphology import remove_small_objects

    return remove_small_objects(im, size_threshold)


def deduce_region_props(im_label):
    # deduce region props
    from skimage.measure import regionprops

    region = regionprops(im_label)

    return region


def calculate_blob_properties(im_label,
                              output_excel_file_string='',
                              output_blob_base_file_string='',
                              im_base=[],
                              display_padding=150):
    # Function analyzes blobs, creating a panda structure and, optionally
    # creating an image for each blob

    from skimage.measure import regionprops
    import pandas as pd
    import warnings

    # Calculate regionprops for the labeled image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        region = regionprops(im_label)

    # Set up for a data dump
    no_of_blobs = len(region)
    blob_data = pd.DataFrame({
                              'label': np.zeros(no_of_blobs),
                              'area': np.zeros(no_of_blobs),
                              'convex_area': np.zeros(no_of_blobs),
                              'eccentricity': np.zeros(no_of_blobs),
                              'equivalent_diameter': np.zeros(no_of_blobs),
                              'extent': np.zeros(no_of_blobs),
                              'major_axis_length': np.zeros(no_of_blobs),
                              'minor_axis_length': np.zeros(no_of_blobs),
                              'orientation': np.zeros(no_of_blobs),
                              'perimeter': np.zeros(no_of_blobs),
                              'solidity': np.zeros(no_of_blobs)})

    for i, r in enumerate(region):
        # Need this bit because of the major and minor axis lengths
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Store blob data in pandas DataFrame
            blob_data.at[i, 'label'] = r.label
            blob_data.at[i, 'area'] = r.area
            blob_data.at[i, 'convex_area'] = r.convex_area
            blob_data.at[i, 'eccentricity'] = r.eccentricity
            blob_data.at[i, 'equivalent_diameter'] = r.equivalent_diameter
            blob_data.at[i, 'extent'] = r.extent
            blob_data.at[i, 'major_axis_length'] = r.major_axis_length
            blob_data.at[i, 'minor_axis_length'] = r.minor_axis_length
            blob_data.at[i, 'orientation'] = r.orientation
            blob_data.at[i, 'perimeter'] = r.perimeter
            blob_data.at[i, 'solidity'] = r.solidity

        if (output_blob_base_file_string):
            # Creates an image showing a padded version of the blob

            # Get the bounding box of the blob
            bbox_coordinates = r.bbox

            # Get the size of im_gray
            rows_cols = im_base.shape

            # Pad the box
            top = np.amax([0, bbox_coordinates[0]-display_padding])
            bottom = np.amin([rows_cols[0], bbox_coordinates[2]+display_padding])
            left = np.amax([0, bbox_coordinates[1]-display_padding])
            right = np.amin([rows_cols[1], bbox_coordinates[3]+display_padding])

            # Create sub_images using the padded box
            im_sub_base = normalize_gray_scale_image(
                    im_base[top:bottom,left:right])
            im_sub_label = im_label[top:bottom,left:right]
            im_mask = np.zeros(im_sub_base.shape)
            im_mask[np.nonzero(im_sub_label == (i+1))] = 1

            # Creates the overlay
            im_b = merge_rgb_planes(np.zeros(im_mask.shape),
                                    np.zeros(im_mask.shape),
                                    im_sub_base)
            im_r = merge_rgb_planes(im_mask,
                                    np.zeros(im_mask.shape),
                                    np.zeros(im_mask.shape))
            im_overlay = np.ubyte(0.5*255*im_b + 0.5*255*im_r)
            im_overlay = im_overlay.astype('uint8')

            # Writes padded blob to an image file created on the fly
            ofs = ('%s_%d.png' % (output_blob_base_file_string, i+1))
            print('Writing blob label %d to %s' % (i+1, ofs))
            write_image_to_file(im_overlay, ofs)

    # Write data to excel if there is a filename
    if output_excel_file_string:
        print('Writing blob data to %s' % output_excel_file_string)

        # Check directory exists and make it if necessary
        dir_path = os.path.dirname(output_excel_file_string)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        blob_data.to_excel(output_excel_file_string)

    # Return blob data
    return blob_data, region


def create_annotated_blob_overlay(im_overlay, regions,
                                  output_file_string,
                                  verbose=1):
    # Adds centroids and blob numbers to an overlay

    from PIL import Image, ImageFont, ImageDraw

    # Write im_overlay to the output file
    write_image_to_file(im_overlay, output_file_string, verbose=0)

    # Now bring it back in using PIL
    im = Image.open(output_file_string)
    font = ImageFont.truetype("arial.ttf", 12)
    d = ImageDraw.Draw(im)

    for i, r in enumerate(regions):
        cent = r['centroid']
        d.text((cent[1], cent[0]), ("%d" % (i + 1)), 'white', font=font)

    if (verbose):
        print('Writing image to %s' % output_file_string)
    im.save(output_file_string)


def shuffle_labeled_image(im_label, bg_color=(1, 1, 1)):
    # Turns a labeled image into an RGB image with blobs with random colors
    # to improve visualization

    from skimage.color import label2rgb
    from matplotlib import cm

    # Get a non-sequential colormap
    color_map = cm.get_cmap('tab20')

    # Create the shuffled image with label2rgb
    # using a white background
    im_shuffle = label2rgb(im_label,
                           colors=color_map.colors,
                           bg_label=0,
                           bg_color=bg_color)

    return im_shuffle


def merge_rgb_planes(im_r,im_g,im_b,weights=[1, 1, 1]):
    # Creates an RGB image from 3 planes

    rows_cols = im_r.shape
    im_out = np.zeros((rows_cols[0], rows_cols[1], 3))

    im_out[:, :, 0] = weights[0] * im_r
    im_out[:, :, 1] = weights[1] * im_g
    im_out[:, :, 2] = weights[2] * im_b

    return im_out


def merge_label_and_blue_image(im_label, im_blue):
    # Places label on top of a blue image

    rows_cols = im_label.shape

    im_out = np.zeros((rows_cols[0], rows_cols[1], 3))
    im_label2 = np.zeros(im_label.shape)
    im_label2[im_label > 0] = 0.5

    im_out[:, :, 0] = im_label2
    im_out[:, :, 2] = im_blue

    return im_out


def correct_background(im):
    # Normalizes background
    from skimage.morphology import reconstruction

    h = 0.5 * np.mean(im)
    seed = im-h
    mask = im
    dilated = reconstruction(seed, mask, method='dilation')

    return dilated


def fill_holes_in_binary_image(im):
    # Fills holes in image

    from scipy.ndimage import binary_fill_holes

    return binary_fill_holes(im)


def fill_holes_in_non_binary_image(im):
    # Fill holes in non_binary image
    from skimage.morphology import reconstruction

    # Save initial type
    ty = im.dtype

    seed = np.copy(im)
    seed[1:-1, 1:-1] = np.amax(im)
    mask = im

    im_filled = reconstruction(seed, mask, method='erosion')

    # Return filled image as the original type
    return im_filled.astype(ty)


def raw_image_file_to_labeled_image(raw_image_file_string,
                                    image_to_label_parameters=[],
                                    results_folder=[],
                                    verbose=1):
    # Code takes an image_file and returns a labeled image

    from skimage.io import imread
    from skimage.util import view_as_blocks

    # Read in the image and convert to float
    if (verbose):
        print('Importing %s as gray-scale' % raw_image_file_string)
    im_gray = imread(raw_image_file_string, as_gray=True)
    if (len(im_gray.shape) > 2):
        print('Image was not read as 2D')
        print('Try forcing conversion')
        im_gray = rgb2gray(im_gray)
    print(im_gray.shape)

    # Get image size
    rows_cols = im_gray.shape

    # If image is smaller than block size, process it in one go
    if (np.amax(rows_cols) <= image_to_label_parameters['block_size']):
        # Process this image
        if (verbose):
            print('Processing image as a single block')

        im_size_filtered, im_sat = \
            process_image_to_blobs(im_gray,
                                   image_to_label_parameters=image_to_label_parameters,
                                   results_folder=results_folder)
    else:
        # We are processing the image as a sequence of blocks
        # First pad the image so that its dimensions are a multiple
        # of the block size

        # Use bs as short hand for block size dictionary entry
        bs = image_to_label_parameters['block_size']

        # Work out the number of blocks we need
        row_blocks = np.ceil(rows_cols[0] / bs).astype(int)
        col_blocks = np.ceil(rows_cols[1] / bs).astype(int)
        row_padding = ((bs * row_blocks) - rows_cols[0]).astype(int)
        col_padding = ((bs * col_blocks) - rows_cols[1]).astype(int)

        # Pad rows and cols with zeros
        im_pad = np.pad(im_gray, [(0, row_padding), (0, col_padding)],
                                  'constant')

        # Create matrices to hold the saturated values and size filtered blobs
        im_sat = np.zeros(im_pad.shape)
        im_size_filtered = np.zeros(im_pad.shape)

        # Process as blocks, each a square of size bs
        pad_blocks = view_as_blocks(im_pad, block_shape=(bs, bs))
        sat_blocks = view_as_blocks(im_sat, block_shape=(bs, bs))
        size_filtered_blocks = view_as_blocks(im_size_filtered, block_shape=(bs, bs))

        # Loop through blocks in a nested loop
        counter = 0
        for i in np.arange(0, row_blocks):
            for j in np.arange(0, col_blocks):

                counter = counter + 1

                if (verbose):
                    print('Processing block %d of %d' %
                          (counter, (row_blocks * col_blocks)))

                size_filtered_blocks[i, j], sat_blocks[i, j] = \
                    process_image_to_blobs(pad_blocks[i, j],
                                           image_to_label_parameters=image_to_label_parameters,
                                           results_folder=results_folder,
                                           block_number=counter)

        # Rebuild images from blocks
        im_sat = sat_blocks.transpose(0, 2, 1, 3). \
            reshape(-1, sat_blocks.shape[1] * sat_blocks.shape[3])
        im_size_filtered = size_filtered_blocks.transpose(0, 2, 1, 3). \
            reshape(-1, size_filtered_blocks.shape[1] * size_filtered_blocks.shape[3])
        im_gray = im_pad

    # We're back on one path
    # (we're done with block processing if we needed to do that)

    # Clear edge and remove small objects
    if (verbose):
        print('Clearing objects from boundaries')
    im_clear_edge = clear_edge(im_size_filtered, invert_mode=0)

    if (verbose):
        print('Labeling image')
    im_label = label_image(im_clear_edge)

    # Tidy up
    return im_label, im_sat, im_gray


def process_image_to_blobs(im_gray,
                           image_to_label_parameters=[],
                           results_folder=[],
                           block_number=1,
                           verbose=1):
    # This is a helper function that takes a gray-scale image and processes
    # it to create blobs

    if (verbose):
        print('Subtracting background')
    im_background_corrected = correct_background(im_gray)

    if (verbose):
        print('Normalizing gray-scale')
    im_norm = normalize_gray_scale_image(im_background_corrected)

    if (verbose):
        print('Saturating image')
    im_sat = saturate_gray_scale_image(
                im_norm,
                image_to_label_parameters['saturation_percent'])

    # Check for image that couldn't be reliably saturated
    if (np.amax(im_sat) == 0):
        if (verbose):
            print('Image could not be saturated so returning zeros')
        im_size_filtered = np.zeros(im_gray.shape)
        return im_size_filtered, im_sat

    # If we get this far, we have some saturated edges
    # Deduce edge
    if (verbose):
        print('Applying Frangi filter')
    im_frangi = apply_Frangi_filter(im_sat)

    if (verbose):
        print('Applying Otsu threshold')
    im_edge = otsu_threshold(im_frangi)

    if (verbose):
        print('Filling holes in thresholded image')
    im_filled = fill_holes_in_binary_image(im_edge)

    if (verbose):
        print('Removing objects below a size of %d' %
                  image_to_label_parameters['min_object_size'])
    im_size_filtered = \
        remove_small_objects(im_filled,
                             image_to_label_parameters['min_object_size'])

    # Try to create a summary image - useful for troublshooting
    if (results_folder):
        block_image_file_string = os.path.join(
                results_folder, 'processing','blocks',
                ('block_%d.png' % block_number))

        if (verbose):
            print('Saving process_image_to_blobs figure to %s' %
                  block_image_file_string)

        fig, ax = plt.subplots(3, 2, figsize=(7, 7))
        ax[0, 0].imshow(im_gray)
        ax[0, 0].set_title('Converted to gray-scale')
        ax[0, 1].imshow(im_background_corrected)
        ax[0, 1].set_title('Background correction')
        ax[1, 0].imshow(im_sat)
        ax[1, 0].set_title('Saturated')
        ax[1, 1].imshow(im_frangi)
        ax[1, 1].set_title('Frangi edges')
        ax[2, 0].imshow(im_edge)
        ax[2, 0].set_title('Otsu threshold')
        ax[2, 1].imshow(im_size_filtered)
        ax[2, 1].set_title('Filled holes')

        # Check directory exists and save image file
        dir_path = os.path.dirname(block_image_file_string)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        plt.savefig(block_image_file_string, bbox_inches='tight')
        plt.close()

    else:
        if (verbose):
            print('No results folder specified - skipping save blocks image')


    return im_size_filtered, im_sat


def handle_potentially_connected_fibers(im_class, im_label,
                                        blob_data, region,
                                        classifier_model,
                                        watershed_distance=10,
                                        troubleshoot_mode=0,
                                        verbose_mode=1):
    # Tries to handle potentially connected fibers

    # Make copies of input arrays
    im_class2 = np.copy(im_class)
    im_label2 = np.copy(im_label)

    # Now create a new image showing only the connected fibers
    im_connected = np.zeros(im_class.shape)
    im_connected[im_class == 2] = im_label[im_class == 2]

    # And of im_connected
    im_connected2 = np.copy(im_connected)

    # Make copy of blob_data
    blob_data2 = blob_data.copy()

    # Make a figure if required for troubleshooting
#    if (troubleshoot_mode):
#        fig, ax = plt.subplots(3, 2, figsize=(7,7))

    # Now loop through im_connected looking for the blobs that require analysis
    blob_counter = np.amax(im_label2)
    for i, r in enumerate(region):
        rc = r.coords
        y = im_connected2[rc[:, 0], rc[:, 1]]
        if (np.any(y == (i + 1))):
            if (verbose_mode):
                print('Potentially connected fiber: %d of %d' % (i, len(region)))

            # Pull off the blob
            # Get the bounding box of the blob
            bbox_coordinates = r.bbox

            top = bbox_coordinates[0]
            bottom = bbox_coordinates[2]
            left = bbox_coordinates[1]
            right = bbox_coordinates[3]

            # Pull off the sub-image containing the connected region
            im_sub_blob = np.copy(im_connected2)[top:bottom, left:right]
            # Make sure that it only contains the connected blob we are
            # currently working on
            im_sub_blob[np.not_equal(im_sub_blob, (i+1))] = 0

            # Pull off the corresponding bit of the labeled image
            im_sub_label = np.copy(im_label2)[top:bottom, left:right]
            # And the class image
            im_sub_class = np.copy(im_class2)[top:bottom, left:right]

            # Get a new labeled image using the watershed algorithm
            im_watershed = apply_watershed(im_sub_blob,
                                           watershed_distance,
                                           troubleshoot_mode)
            max_watershed = np.amax(im_watershed)

#            if (i==128):
#                f,ax = plt.subplots(3,2, figsize=(8,8))
#                ax[0,0].imshow(im_sub_blob)
#                ax[0,1].imshow(im_sub_class)
#                ax[1,0].imshow(im_watershed)

            if (max_watershed > 0):
                # There was at least one blob
                # Classify them to get new properties
                im_class_new, blob_data_new, temp = \
                    ml.classify_labeled_image(im_watershed, classifier_model)

                # Update the sub_label with new labels
                im_bw = np.zeros(im_watershed.shape)
                im_bw[im_watershed > 0] = 1
                im_sub_label[im_sub_blob > 0] = \
                    im_watershed[im_sub_blob > 0]
                im_sub_label = im_sub_label + \
                    (blob_counter * im_bw)
                im_label2[top:bottom, left:right] = im_sub_label

                # Similarly, update im_sub_class
                im_sub_class[im_sub_blob > 0] = 0
                im_sub_class[im_sub_blob > 0] = im_class_new[im_sub_blob > 0]

                # Similarly substitute new class labels
                im_class2[top:bottom, left:right] = im_sub_class

                # Add new blobs to blob_data
                blob_data2.append(blob_data_new)

                # Update blob_counter
                blob_counter = blob_counter + np.amax(im_watershed)
            else:
                print('Watershed discarded all potential blobs')

#
#            if (troubleshoot_mode):
#                ax[0, 0].imshow(im_sub_blob)
#                ax[0, 0].set_title('im_sub_blob')
#                ax[0, 1].imshow(im_sub_label)
#                ax[0, 1].set_title('im_sub_label')
#                ax[1, 0].imshow(im_watershed)
#                ax[1, 0].set_title('im_watershed')
#                ax[1, 1].imshow(im_sub_class)
#                ax[1, 1].set_title('im_sub_class')

    return im_class2, im_label2


def apply_watershed(im_blob, max_size, troubleshoot_mode=0):
    # Applies the watershed algorithm in an attempt to separate fibers
    # returns a new labeld image
    # Code is based on this example: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed, erosion

    # Copy im_blob
    im_blob2 = np.copy(im_blob)

    # Make it a binary image
    im_blob2[im_blob2 > 0] = 1

    # Pad
    im_blob2 = np.pad(im_blob2, [[1, 1], [1, 1]], 'constant')

    im_dist = ndi.distance_transform_edt(im_blob2)
    im_dist[im_dist >= max_size] = max_size
    im_peaks = peak_local_max(im_dist, indices=False,
                              labels=im_blob2)
    im_peaks = remove_small_objects(im_peaks, max_size)
    im_peaks = label_image(im_peaks)

    im_label = watershed(-im_dist, im_peaks, mask=im_blob2,
                         watershed_line=True)
    
    # Erode image to separate blobs
    im_label = erosion(im_label)

    # Return to original size to account for padding
    im_label = im_label[1:-1, 1:-1]

#    if (troubleshoot_mode):
#        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(5,10),nrows=4)
#        ax1.imshow(im_blob)
#        ax1.set_title("im_blob")
#        p2=ax2.imshow(im_dist)
#        ax2.set_title("im_dist")
#        fig.colorbar(p2,ax=ax2)
#        p3=ax3.imshow(im_peaks)
#        fig.colorbar(p3,ax=ax3)
#        p4=ax4.imshow(im_label)

    # Return re-imaged label
    return im_label


def refine_fiber_edges(im_seeds, im_gray,
                       refine_fibers_parameters=[]):
    # Refines fiber edges using an algorithm from scikit-image

    from skimage.segmentation import (morphological_geodesic_active_contour,
                                      inverse_gaussian_gradient)

    # Calculate the gradient image
    im_gradient = inverse_gaussian_gradient(im_gray,
                                            sigma = refine_fibers_parameters['sigma'])

    # Now do the fitting
    im_out = morphological_geodesic_active_contour(im_gradient,
                                        refine_fibers_parameters['max_iterations'],
                                        init_level_set=im_seeds)

    return im_out, im_gradient
