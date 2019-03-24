import numpy as np
from PIL import Image


# START im_overlay*****************************************************
def im_overlay(im1, im2, alpha=0.5):

    if isinstance(im1, np.ndarray):
        temp_im1 = (im1 * 255).astype(np.uint8)
        pil_im1 = Image.fromarray(temp_im1)
    else:
        pil_im1 = im1

    if isinstance(im2, np.ndarray):
        temp_im2 = (im2 * 255).astype(np.uint8)
        pil_im2 = Image.fromarray(temp_im2)
    else:
        pil_im2 = im2

    if im1.dtype == 'bool':
        overlay = pil_im2
        overlay.paste((0,255,255), mask=pil_im1)
    elif im2.dtype == 'bool':
        overlay = im1
        overlay.paste(overlay, mask=im2)
    else:
        overlay = Image.blend(pil_im1, pil_im2, alpha)

    return overlay
