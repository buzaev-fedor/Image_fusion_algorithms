import cv2
import numpy as np
from skimage.transform import rescale


def resize_image(multispectral_image, panchromatic_image):
    dim = (panchromatic_image.shape[1], panchromatic_image.shape[0])
    resized_image = cv2.resize(multispectral_image, dim,
                               interpolation=cv2.INTER_CUBIC).astype(type(multispectral_image))
    return resized_image


def stretch(bands, lower_line=2, higher_line=98):
    out = np.zeros_like(bands)
    for i in range(3):
        lower_percentile = np.percentile(bands[:, :, i], lower_line)
        higher_percentile = np.percentile(bands[:, :, i], higher_line)
        result_coordinates = (bands[:, :, i] - lower_percentile) * 255 / (higher_percentile - lower_percentile)
        result_coordinates[result_coordinates < 0] = 0
        result_coordinates[result_coordinates > 255] = 255
        out[:, :, i] = result_coordinates
    return out.astype(np.uint8)


def prepare_scaled_rgbn_image(multispectral):
    # get multispectral bands
    rgbn = np.empty((multispectral.shape[0], multispectral.shape[1], 4))

    rgbn[:, :, 2] = multispectral[:, :, 2]  # red
    rgbn[:, :, 1] = multispectral[:, :, 1]  # green
    rgbn[:, :, 0] = multispectral[:, :, 0]  # blue
    rgbn[:, :, 3] = multispectral[:, :, 3]  # NIR-1

    # Scaled them
    rgbn_scaled = np.empty((multispectral.shape[0] * 3, multispectral.shape[1] * 3, 4))

    for i in range(4):
        tmp_image = rgbn[:, :, i]
        scaled = rescale(tmp_image, (3, 3))
        rgbn_scaled[:, :, i] = scaled

    return rgbn_scaled


def check_and_crop(panchromatic, rgbn_scaled):
    # check size and crop for panchromatic band
    if panchromatic.shape[0] < rgbn_scaled.shape[0]:
        rgbn_scaled = rgbn_scaled[:panchromatic.shape[0], :, :]
    else:
        panchromatic = panchromatic[:rgbn_scaled.shape[0], :]
    if panchromatic.shape[1] < rgbn_scaled.shape[1]:
        rgbn_scaled = rgbn_scaled[:, :panchromatic.shape[1], :]
    else:
        panchromatic = panchromatic[:, :rgbn_scaled.shape[1]]

    # panchromatic = np.mean(panchromatic, axis=1)
    return panchromatic, rgbn_scaled
