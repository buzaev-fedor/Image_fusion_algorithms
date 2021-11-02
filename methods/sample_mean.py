import numpy as np


def sample_mean(panchromatic_image, rgbn_scaled):
    red_channel = 0.5 * (rgbn_scaled[:, :, 0] + panchromatic_image)[:, :, np.newaxis]
    green_channel = 0.5 * (rgbn_scaled[:, :, 1] + panchromatic_image)[:, :, np.newaxis]
    blue_channel = 0.5 * (rgbn_scaled[:, :, 2] + panchromatic_image)[:, :, np.newaxis]

    image = np.concatenate([red_channel, green_channel, blue_channel], axis=2)
    return image
