# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:53:42 2019

@author: kscamp3
"""
import scipy as sp
import numpy as np
from skimage.color import rgb2gray
from skimage.util import invert
import matplotlib.pyplot as plt

try:
    import py_vision.machine_learing.machine_learn as ml
except:
    import sys
    sys.path.append('C:\ken\GitHub\CampbellMuscleLab\Projects\Python_MyoVision\kens_tinkering\Python_code')
    import machine_learning.machine_learn as ml

def kens_test():
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import rgb2gray
    from skimage import data
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour
    
    
    img = data.astronaut()
    img = rgb2gray(img)
    
    s = np.linspace(0, 2*np.pi, 400)
    x = 220 + 100*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    print(init)
    
    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

def return_gray_scale_image(rgb_image):
    # converts a color image to a gray-scale image
    return rgb2gray(rgb_image)

def normalize_gray_scale_image(gray_image):
    # normalizes a gray-scale image between 0 and 1
    norm_image = gray_image - np.amin(gray_image)
    norm_image = norm_image / np.amax(norm_image)
    return norm_image

def saturate_gray_scale_image(gray_image, x):
    # saturates a gray-scale image so that x% of pixels are at low or high
    from skimage.exposure import rescale_intensity
    
    lo, hi = np.percentile(gray_image, (x, 100-x))
    sat_im = rescale_intensity(gray_image, in_range=(lo, hi))
    return sat_im

def apply_Frangi_filter(im):
    # applies a Frangi filter to detect edges
    from skimage.filters import frangi

    im = 255.0 * im

    im_out = frangi(invert(im),
                    scale_range = (1, 4))

    return im_out

def otsu_threshold(gray_image):
    # implements an otsu threshold
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(gray_image)
    im_binary = gray_image > thresh
    im_binary = im_binary.astype(sp.bool_)
    im_binary = invert(im_binary)
    
    return im_binary

def label_image(im):
    from skimage.measure import label

    im_label = label(im)
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
    z =im.reshape((-1,1))
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
               output_image_base_file_string="", display_padding=20, im_gray = [],
               output_excel_file_string=""):
    # Function analyzes blobs, creating a panda structure and, optionally
    # creating an image for each blob
    
    import matplotlib.pyplot as plt
    from skimage.measure import regionprops
    from skimage.color import label2rgb
    from skimage.io import imsave
    import pandas as pd

    # Calculate regionprops for the labeled image
    region = regionprops(im_label)

    # Set up for a data dump
    no_of_blobs = len(region)
    blob_data = pd.DataFrame({
                              'label' : np.zeros(no_of_blobs),
                              'area' : np.zeros(no_of_blobs),
                              'eccentricity': np.zeros(no_of_blobs),
                              'convex_area': np.zeros(no_of_blobs),
                              'equivalent_diameter': np.zeros(no_of_blobs),
                              'extent': np.zeros(no_of_blobs),
                              'major_axis_length': np.zeros(no_of_blobs),
                              'minor_axis_length': np.zeros(no_of_blobs),
                              'solidity': np.zeros(no_of_blobs)})
#
#                              'centroid_row': np.zeros(no_of_blobs),
#                              'centroid_col': np.zeros(no_of_blobs),
#                              'euler_number': np.zeros(no_of_blobs),
    
    for i,r in enumerate(region):

        # Store blob data in pandas DataFrame
        blob_data.at[i, 'label'] = r.label
        blob_data.at[i, 'area'] = r.area
        blob_data.at[i, 'eccentricity'] = r.eccentricity
#        blob_data.at[i, 'centroid_row'] = r.centroid[0]
#        blob_data.at[i, 'centroid_col'] = r.centroid[1]
        blob_data.at[i, 'convex_area'] = r.convex_area
        blob_data.at[i, 'equivalent_diameter'] = r.equivalent_diameter
#        blob_data.at[i, 'euler_number'] = r.euler_number
        blob_data.at[i, 'extent'] = r.extent
        blob_data.at[i, 'major_axis_length'] = r.major_axis_length
        blob_data.at[i, 'minor_axis_length'] = r.minor_axis_length
        blob_data.at[i, 'solidity'] = r.solidity

        if (output_image_base_file_string):
            # Creates an image showing a padded version of the blob

            # Get the bounding box of the blob
            bbox_coordinates = r.bbox

            # Get the size of im_gray
            rows_cols = im_gray.shape

            # Pad the box
            top = np.amax([0, bbox_coordinates[0]-display_padding])
            bottom = np.amin([rows_cols[0], bbox_coordinates[2]+display_padding])
            left = np.amax([0, bbox_coordinates[1]-display_padding])
            right = np.amin([rows_cols[1], bbox_coordinates[3]+display_padding])

            # Create sub_images using the padded box
            im_sub_gray = im_gray[top:bottom,left:right]
            im_sub_label = im_label[top:bottom,left:right]
            im_mask = np.zeros(im_sub_gray.shape)
            im_mask[np.nonzero(im_sub_label == (i+1))] = 1

            # Creates the overlay
            im_overlay = label2rgb(im_mask, im_sub_gray, alpha = 0.3)

            # Writes padded blob to an image file created on the fly
            ofs = ('%s_%d.png' % (output_image_base_file_string,i+1))
            print('Writing blob label %d to %s' % (i+1, ofs))
            imsave(ofs,im_overlay)

#        if (i==3):
#            break

    # Write data to excel
    if (output_excel_file_string):
        print('Writing blob data to %s' % output_excel_file_string)
        blob_data.to_excel(output_excel_file_string)

    # Return blob data
    return blob_data, region

def shuffle_labeled_image(im_label):
    # Turns a labeled image into an RGB image with blobs with random colors
    # to improve visualization

    # Find the number of labels
    no_of_labels = np.amax(im_label)

    # Set up random colors
    random_color = np.random.rand(3, no_of_labels)

    # Create im_shuffle as white matrix
    s = im_label.shape
    im_shuffle = np.ones((s[0], s[1], 3))

    for i in np.arange(1, no_of_labels+1):
        # Do an index to r,c for each label
        vi = np.nonzero(im_label == i)
        r = vi[0]
        c = vi[1]
        for j in np.arange(0, len(r)):
            im_shuffle[r[j], c[j], :] = random_color[:, i-1]

    return im_shuffle

def raw_image_file_to_labeled_image(raw_image_file_string,
                                saturation_percent = 5,
                                min_object_size = 5):
    # Code takes an image_file and returns a labeled image 

    # Read in the image
    import cv2
    im = cv2.imread(raw_image_file_string)
    
    # Convert to gray-scale, normalize, and saturate
    im_gray = return_gray_scale_image(im)
    im_norm = normalize_gray_scale_image(im_gray)
    im_sat = saturate_gray_scale_image(im_norm, saturation_percent)
    
    # Deduce edge
    im_frangi = apply_Frangi_filter(im_sat)
    im_edge = otsu_threshold(im_frangi)

    # Clear edge and remove small objects
    im_clear_edge = clear_edge(im_edge, invert_mode = 0)
    im_remove_small_objects = remove_small_objects(im_clear_edge,
                                                   min_object_size)

    # Label image
    im_label = label_image(im_remove_small_objects)

    # Tidy up
    return im_label, im_sat;

def handle_potentially_connected_fibers(im_class, im_label,
                                        blob_data, region,
                                        classifier_model,
                                        troubleshoot_mode=0):
    # Tries to handle potentially connected fibers

    # Make copies of input arrays
    im_class2 = np.copy(im_class)
    im_label2 = np.copy(im_label)

    # First create a new image showing only the connected fibers
    im_connected = np.zeros(im_class.shape)
    im_connected[im_class == 2] = im_label[im_class == 2]

    # And of im_connected
    im_connected2 = np.copy(im_connected)
    
    # Make copy of blob_data
    blob_data2 = blob_data.copy()

    # Now loop through im_connected looking for the blobs that require analysis
    blob_counter = np.amax(im_label2)
    for i, r in enumerate(region):
        if (np.any(im_connected2 == (i+1))):
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
            im_sub_blob[np.not_equal(im_sub_blob, (i+1))]=0

            # Pull off the corresponding bit of the labeled image
            im_sub_label = np.copy(im_label2)[top:bottom, left:right]
            # And the class image
            im_sub_class = np.copy(im_class2)[top:bottom, left:right]

            # Get a new labeled image using the watershed algorithm
            im_watershed = apply_watershed(im_sub_blob, 5, troubleshoot_mode)
            max_watershed = np.amax(im_watershed)

            # Classify that to get new properties
            im_class_new, blob_data_new = \
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
            
#            if (troubleshoot_mode):
#                fig, ax = plt.subplots(3, 2, figsize=(7,7))
#                ax[0, 0].imshow(im_sub_blob)
#                ax[0, 0].set_title('im_sub_blob')
#                ax[0, 1].imshow(im_sub_label)
#                ax[0, 1].set_title('im_sub_label')
#                ax[1, 0].imshow(im_watershed)
#                ax[1, 0].set_title('im_watershed')
#                ax[1, 1].imshow(im_sub_class)
#                ax[1, 1].set_title('im_sub_class')
#                break

    return im_class2, im_label2

#    fig, (ax1, ax2) = plt.subplots(figsize=(5,5), nrows=2)
#    p = ax1.imshow(im_connected)
#    fig.colorbar(p,ax=ax1)
#    ax2.imshow(im_blob)
    
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

    if (troubleshoot_mode):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(5,10),nrows=4)
        ax1.imshow(im_blob)
        ax1.set_title("im_blob")
        p2=ax2.imshow(im_dist)
        ax2.set_title("im_dist")
        fig.colorbar(p2,ax=ax2)
        p3=ax3.imshow(im_peaks)
        fig.colorbar(p3,ax=ax3)
        p4=ax4.imshow(im_label)

    # Return re-imaged label
    return im_label

def refine_fiber_edges(im_class, im_sat, refine_padding = 10,
                       troubleshoot_mode=0):
    # Refines fiber edges

    from skimage.measure import regionprops
    from skimage.measure import find_contours
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    from skimage.segmentation import active_contour
    from skimage.draw import polygon
    from scipy import ndimage as ndi
    

    # Create a new image showing fibers
    im_fibers = np.zeros(im_class.shape)
    im_fibers[im_class == 1] = 1

    # Label it
    im_label = label_image(im_fibers)
    
    # Get copy of im_class
    im_class2 = np.copy(im_class)
    
    # Copy the im_sat
    im_sat2 = np.copy(im_sat)
    
    fig,ax = plt.subplots(1,1)
    ax.imshow(im_label)
    
    no_of_fibers = np.amax(im_label)
    
    # Calculate regionprops for the labeled image
    region = regionprops(im_label)
    
    rows_cols = im_sat.shape
    
    # Create image result
    im_final = np.zeros(im_label.shape)

    # Cycle through them
    for i, r in enumerate(region):
        print("Refining fiber %d of %d" % (i, no_of_fibers))

        # Pull off the blob
        # Get the bounding box of the blob
        bbox_coordinates = r.bbox

        top = np.amax([0, bbox_coordinates[0]-refine_padding])
        bottom = np.amin([rows_cols[0], bbox_coordinates[2]+refine_padding])
        left = np.amax([0, bbox_coordinates[1]-refine_padding])
        right = np.amin([rows_cols[1], bbox_coordinates[3]+refine_padding])

        # Pull off the sub-image containing the connected region
        im_blob = np.copy(im_label)[top:bottom, left:right]
#        im_blob = np.pad(im_blob, [[1, 1], [1, 1]], 'constant')
        # Make sure that it only contains the connected blob we are
        # currently working on
        im_blob[np.not_equal(im_blob, (i+1))]=0
        im_blob[im_blob>0]=1
#        
#        fig,ax = plt.subplots(1,1)
#        ax.imshow(im_blob)

        # Get boundaries as coordinates
        # First if there are many
        xx = find_contours(im_blob,0.5)[0]
        snake_init = np.squeeze(xx)[:, ::-1]
#        print(snake_init)

        # Pull off the raw image
        im_raw = np.copy(im_sat2)[top:bottom, left:right]
        # Pad that
#        im_raw = np.pad(im_raw, [[1, 1], [1, 1]], 'edge')
        
        # First overlay
        im_mask = np.zeros(im_blob.shape)
        im_mask[im_blob>0] = 1
        im_overlay = label2rgb(im_mask, im_raw)
#        im_overlay2 = label2rgb(im_overlay, im_boundaries)

        # Apply active countour
        snake_final = active_contour(im_raw, snake_init,
                                     beta=10)

        # Find the points in the snake
        im_result = np.zeros(im_blob.shape)
        rr,cc = polygon(snake_final[:, 1], snake_final[:, 0])
        im_result[rr, cc] = 1

        im_shuffle = shuffle_labeled_image(im_label)
        im_overlay2 = label2rgb(im_result, im_raw)
        
        im_final[top:bottom, left:right] = \
            im_final[top:bottom, left:right] + (i+1)*im_result
        
        if (troubleshoot_mode):
            fig, ax = plt.subplots(4,2, figsize=(10,7))
            ax[0, 0].imshow(im_shuffle)
            ax[0, 1].imshow(im_class2)
            ax[1, 0].imshow(im_blob)
            ax[1, 1].imshow(im_raw)
            ax[2, 0].imshow(im_overlay)
            ax[2, 0].plot(snake_init[:, 0], snake_init[:, 1],'-g', lw=2)
            ax[2, 1].imshow(im_raw)
            ax[2, 1].plot(snake_final[:, 0], snake_final[:, 1],'-r', lw=2)
            ax[3, 0].imshow(im_result)
            ax[3, 1].imshow(im_overlay2)
            
            break
        
    return im_final