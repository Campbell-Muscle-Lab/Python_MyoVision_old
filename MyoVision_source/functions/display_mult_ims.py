import matplotlib.pyplot as plt
import scipy as sp
from tkinter import *
import math
import copy
import re
import gc

# START display_mult_ims*******************************************************
# Description:
#     Displays all given images in the same figure either with linked axes or not
# Inputs:
#     *ims: Any number of RGB or RGBA images specified by ndarrays
#     shareaxes: Boolean specifying whether to link the image shareaxes
# Returns:
#     N/A
def display_mult_ims(*ims, shareaxes=False):
    fig = plt.figure()
    # Calculates the best number of rows and columns for the number of images given
    columns = math.ceil(math.sqrt(len(ims)))
    rows = math.ceil(len(ims)/columns)
    count = 0
    for im in ims:
        fig.add_subplot(rows, columns, (count+1))
        # Defaults to grayscale colormap for image slices
        im_colormap = None
        # if len(im.shape) == 2:
        #     im_colormap = "gray"
        # Links subplot axes (zoom and pan them together)
        if (count > 0 and shareaxes):
            prev_ax = fig.axes[count-1]
            fig.axes[count].get_shared_x_axes().join(fig.axes[count], prev_ax)
            fig.axes[count].get_shared_y_axes().join(fig.axes[count], prev_ax)
        plt.imshow(im, cmap=im_colormap)
        count += 1

    plt.show()
# END display_mult_ims/////////////////////////////////////////////////////////


# START blob_selector**********************************************************
def blob_selector(label_im, underlayed_im=None):

    fig = plt.figure()

    def show_blob(blob, fig):
        plt.figure(fig.number)
        base = copy.deepcopy(underlayed_im)
        blobs_to_show = sp.empty((0))
        try:
            groups = blob.split(',')
            for group in groups:
                # regex help from:  https: // stackoverflow.com / questions / 33225900 / find - all - numbers - in -a - string - in -python - 3?noredirect = 1 & lq = 1
                rng = re.findall(r'\d+', group)
                if len(rng) > 1:
                    low_blob = min(int(rng[0]), int(rng[1]))
                    high_blob = max(int(rng[0]), int(rng[1]))
                else:
                    low_blob = int(rng[0])
                    high_blob = low_blob
                blobs_to_show = sp.append(blobs_to_show, sp.arange(low_blob, high_blob+1))
                # print(blobs_to_show)
        except ValueError:
            print("Must enter an integer")
            return

        mask = sp.isin(label_im, (blobs_to_show+1))
        base[mask, :] = [1,1,1]
        plt.imshow(base)
        plt.show()

    print('Enter blob number or range (q to quit):')
    blob = input()

    while blob is not 'q':
        show_blob(blob, fig)
        print('Enter blob number or range (q to quit):')
        blob = input()
# END blob_selector////////////////////////////////////////////////////////////