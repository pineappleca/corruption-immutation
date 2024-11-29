#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:高斯噪声.py
@time:2024/11/17 22:28:44
@author:Yao Yongrui
'''

from Camera_corruptions import ImageAddGaussianNoise
from Camera_corruptions import ImageAddImpulseNoise
import cv2
import os

image_filename = './n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460.jpg'

gn_output_dir = './gaussian_noise_plot'
if not os.path.exists(gn_output_dir):
    os.makedirs(gn_output_dir)

in_output_dir = './impulse_noise_plot'
if not os.path.exists(in_output_dir):
    os.makedirs(in_output_dir)

def gaussian_noise_plot(image_filename, severity):
    '''
    高斯噪声
    '''
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageAddGaussianNoise(severity=severity, seed=2022)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(gn_output_dir, f'0{severity}.jpg'), image_bgr_blur)

def impulse_noise_plot(image_filename, severity):
    '''
    脉冲噪声
    '''
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageAddImpulseNoise(severity=severity, seed=2022)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(in_output_dir, f'0{severity}.jpg'), image_bgr_blur)

if __name__ == '__main__':
    for severity in range(1, 6):
        gaussian_noise_plot(image_filename, severity)
        impulse_noise_plot(image_filename, severity)