from utils import *
from methods.broovey import *
from methods.hsv_esri import *
from methods.sample_mean import sample_mean

import cv2
import numpy as np
import tifffile as tiff
import PySimpleGUI as sg

sg.theme('DarkAmber')
layout = [
    [sg.Text('Algorithms for a image fusion', size=(22, 1), font=('Any', 18), text_color='#1c86ee',
             justification='left')],
    [sg.Text('Path to input panchromatic image'), sg.In(size=(40, 1), key='input_pan'), sg.FileBrowse()],
    [sg.Text('Path to input multispectral image'), sg.In(size=(40, 1), key='input_ms'), sg.FileBrowse()],
    [sg.Text(' ' * 8), sg.Checkbox('Sample Broovey', key='_SAMPLE_BROOVEY_')],
    [sg.Text(' ' * 8), sg.Checkbox('Broovey', key='_BROOVEY_')],
    [sg.Text(' ' * 8), sg.Checkbox('Sample mean fusion', key='_SIMPLE_MEAN_')],
    [sg.Text(' ' * 8), sg.Checkbox('ESRI fusion', key='_ESRI_')],
    [sg.Text(' ' * 8), sg.Checkbox('HSV fusion', key='_HSV_')],

    [sg.Text('Weight'),
     sg.Slider(range=(0.01, 1), orientation='h', resolution=0.01, default_value=0.1, size=(15, 15), key='weight'),
     sg.T('  ', key='_W_OUT_')],

    [sg.OK(), sg.Cancel(), sg.Stretch()],
]

win = sg.Window('Algorithms for a image fusion',
                default_element_size=(25, 1),
                text_justification='right',
                auto_size_text=True).Layout(layout)
event, values = win.Read()

# panchromatic_image = tiff.imread(values.get("input_pan"))
# multispectral_image = tiff.imread(values.get("input_ms"))
#
panchromatic_image = tiff.imread('/users/buzaev/desktop/test_images/pan_img.tif')
multispectral_image = tiff.imread('/users/buzaev/desktop/test_images/ms_img.tif')

weight = values.get("weight")

USE_SIMPLE_BROOVEY = values.get("_SAMPLE_BROOVEY_")
USE_BROOVEY = values.get("_BROOVEY_")
USE_SAMPLE_MEAN = values.get("_SIMPLE_MEAN_")
USE_ESRI = values.get("_ESRI_")
USE_HSV = values.get("_HSV_")

rgbn_scaled = prepare_scaled_rgbn_image(multispectral_image)
print(rgbn_scaled.shape)
panchromatic_image, rgbn_scaled = check_and_crop(panchromatic_image, rgbn_scaled)
print(type(panchromatic_image))

if USE_SIMPLE_BROOVEY:
    result_image_simple_broovey = simple_broovey(panchromatic_image, rgbn_scaled)
    cv2.imshow("simple_broovey", stretch(result_image_simple_broovey))
    cv2.imwrite("./results_images/simple_broovey.tif", stretch(result_image_simple_broovey))

if USE_BROOVEY:
    result_image_broovey = brovey(panchromatic_image, rgbn_scaled, weight)
    cv2.imshow("broovey", stretch(result_image_broovey))
    cv2.imwrite("./results_images/broovey.tif", stretch(result_image_broovey))

if USE_SAMPLE_MEAN:
    result_image_sample_mean = sample_mean(panchromatic_image, rgbn_scaled)
    cv2.imshow("sample_mean", stretch(result_image_sample_mean))
    cv2.imwrite("./results_images/sample_mean.tif", stretch(result_image_sample_mean))

if USE_ESRI:
    result_image_esri = esri(panchromatic_image, rgbn_scaled)
    cv2.imshow("esri", stretch(result_image_esri))
    cv2.imwrite("./results_images/esri.tif", stretch(result_image_esri))

if USE_HSV:
    result_image_hsv = hsv(panchromatic_image, rgbn_scaled, weight)
    cv2.imshow("hsv", stretch(result_image_hsv))
    cv2.imwrite("./results_images/hsv.tif", stretch(result_image_hsv))

#
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

#
win.close()
