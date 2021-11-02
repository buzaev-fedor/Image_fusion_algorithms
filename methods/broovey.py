import numpy as np


def simple_broovey(panchromatic_image, rgbn_scaled):
    all_in = rgbn_scaled[:, :, 0] + rgbn_scaled[:, :, 1] + rgbn_scaled[:, :, 2]

    red_channel = np.multiply(rgbn_scaled[:, :, 0], panchromatic_image / all_in)[:, :, np.newaxis]
    green_channel = np.multiply(rgbn_scaled[:, :, 1], panchromatic_image / all_in)[:, :, np.newaxis]
    blue_channel = np.multiply(rgbn_scaled[:, :, 2], panchromatic_image / all_in)[:, :, np.newaxis]

    image = np.concatenate([red_channel, green_channel, blue_channel], axis=2)

    return image


def brovey(panchromatic_image, rgbn_scaled, weight=0.1):
    DNF = (panchromatic_image - weight * rgbn_scaled[:, :, 3]) / (
            weight * rgbn_scaled[:, :, 0] + weight * rgbn_scaled[:, :, 1] + weight * rgbn_scaled[:, :, 2])

    red_channel = (rgbn_scaled[:, :, 0] * DNF)[:, :, np.newaxis]
    green_channel = (rgbn_scaled[:, :, 1] * DNF)[:, :, np.newaxis]
    blue_channel = (rgbn_scaled[:, :, 2] * DNF)[:, :, np.newaxis]
    i_channel = (rgbn_scaled[:, :, 3] * DNF)[:, :, np.newaxis]

    image = np.concatenate([red_channel, green_channel, blue_channel, i_channel], axis=2)
    return image
