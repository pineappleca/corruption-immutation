#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:极端天气.py
@time:2024/11/17 22:46:20
@author:Yao Yongrui
'''

from Camera_corruptions import ImageAddSnow, ImageAddRain, ImageAddFog
import cv2
import os

image_filename = './n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460.jpg'

snow_output_dir = './snow_gen_plot'
if not os.path.exists(snow_output_dir):
    os.makedirs(snow_output_dir)

rain_output_dir = './rain_gen_plot'
if not os.path.exists(rain_output_dir):
    os.makedirs(rain_output_dir)

fog_output_dir = './fog_gen_plot'
if not os.path.exists(fog_output_dir):
    os.makedirs(fog_output_dir)

def snow_plot(image_filename, severity):
    '''
    下雪
    '''
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageAddSnow(severity=severity, seed=2022)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(snow_output_dir, f'0{severity}.jpg'), image_bgr_blur)

def rain_plot(image_filename, severity):
    '''
    下雨
    '''
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageAddRain(severity=severity, seed=2022)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(rain_output_dir, f'0{severity}.jpg'), image_bgr_blur)

def fog_plot(image_filename, severity):
    '''
    雾天
    '''
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageAddFog(severity=severity, seed=2022)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(fog_output_dir, f'0{severity}.jpg'), image_bgr_blur)

if __name__ == '__main__':
    for severity in range(1, 6):
        snow_plot(image_filename, severity)
        rain_plot(image_filename, severity)
        fog_plot(image_filename, severity)