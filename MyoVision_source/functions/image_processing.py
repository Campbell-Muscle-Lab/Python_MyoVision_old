import scipy as sp
from PIL import Image
from skimage.morphology import reconstruction, local_minima, local_maxima
from skimage.util import invert
from functions.display_mult_ims import display_mult_ims
import cv2


def print_numpy_image(image):
    print("PRINTING IMAGE INFORMATION")
    print("\tShape: ", image.shape)
    print("\tData Type: ", image.dtype)
    print("\tRange: ", sp.amax(image)-sp.amin(image))
    print("\t\tMax Value: ", sp.amax(image))
    print("\t\tMin Value: ", sp.amin(image))


def numpy_binary_to_pil(numpy_array):
    # Next 3 lines from https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
    size = numpy_array.shape[::-1]
    databytes = sp.packbits(numpy_array, axis=1)
    pil_im = Image.frombytes(mode='1', size=size, data=databytes)
    pil_im = pil_im.convert('RGB')

    return pil_im

def numpy_rgb_to_pil(numpy_array):
    # Next line from https://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
    return Image.fromarray(sp.uint8(numpy_array*255))


# Based on MATLAB 2018a code by The MathWorks Inc.
#     Referenced Function: imhmax
def im_h_max(image, h):
    rec = reconstruction(cv2.subtract(image, h), image)
    return rec


# Based on MATLAB 2018a code by The MathWorks Inc.
#     Referenced Function: imhmin
def im_h_min(image, h):
    temp_inv = invert(image, signed_float=False)
    rec = reconstruction(cv2.subtract(temp_inv, h), temp_inv)
    rec = invert(rec, signed_float=False)
    return rec


# Based on MATLAB 2018a code by The MathWorks Inc.
#     Referenced Function: imextendedmax
def im_extended_max(image, h):
    return local_maxima(im_h_max(image, h))


# Based on MATLAB 2018a code by The MathWorks Inc.
#     Referenced Function: imextendedmin
def im_extended_min(image, h):
    return local_minima(im_h_min(image, h))


# Based on MATLAB 2018a code by The MathWorks Inc.
#     Referenced Function: imimposemin
def im_impose_min(image, mask):
    mask = mask.astype(bool)

    temp_mask = image.astype('float32')
    temp_mask[mask] = 0
    temp_mask[invert(mask)] = 1

    if sp.issubdtype(sp.float64, image.dtype) or sp.issubdtype(sp.float32, image.dtype):
        print("here")
        range = float(sp.amax(image)) - float(sp.amin(image))
        if range == 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        print("there")
        h = 1

    fp1 = (image + h)

    some = sp.minimum(fp1, temp_mask)

    imposed = reconstruction(invert(temp_mask), invert(some))
    imposed = invert(imposed)

    return imposed

