""" 
Python implementation of the color map function for the PASCAL VOC data set: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

Fo
"""
import numpy as np
from PIL import Image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def vis_mask(image, mask, alpha=1):
    '''
        image: PIL Image
        mask: predicted or ground truth mask (containing the labels for the 20+1 classes)
        alpha: blending the image and the mask 
    '''
    cmap = color_map()[:, np.newaxis, :]
    new_im = np.dot(mask == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(mask == i, cmap[i])
    new_im = Image.fromarray(new_im.astype(np.uint8))
    blend_image = Image.blend(image, new_im, alpha=alpha)
    blend_image.save('rgb-mask.jpg')