import numpy as np
import skimage.color as color


def esri(panchromatic_image, rgbn_scaled):
    ADJ = panchromatic_image - rgbn_scaled.mean(axis=2)

    red_channel = (rgbn_scaled[:, :, 0] + ADJ)[:, :, np.newaxis]
    green_channel = (rgbn_scaled[:, :, 1] + ADJ)[:, :, np.newaxis]
    blue_channel = (rgbn_scaled[:, :, 2] + ADJ)[:, :, np.newaxis]
    i = (rgbn_scaled[:, :, 3] + ADJ)[:, :, np.newaxis]

    image = np.concatenate([red_channel, green_channel, blue_channel, i], axis=2)
    return image


def hsv(panchromatic_image, rgbn_scaled, weight):
    hsv = color.rgb2hsv(rgbn_scaled[:, :, :3])
    hsv[:, :, 2] = panchromatic_image - rgbn_scaled[:, :, 3] * weight
    image = color.hsv2rgb(hsv)
    return image
