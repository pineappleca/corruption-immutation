#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:镜头模糊.py
@time:2024/11/17 22:00:13
@author:Yao Yongrui
'''

from Camera_corruptions import ImageMotionBlurFrontBack
import cv2
import os

image_filename = './nuscenes_example_pic/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg'
output_dir = './cam_blur_plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def image_blur_plot(image_filename, severity):
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imbf = ImageMotionBlurFrontBack(severity=severity)
    image_rgb_blur = imbf(image_rgb)
    image_bgr_blur = cv2.cvtColor(image_rgb_blur, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f'0{severity}.jpg'), image_bgr_blur)

if __name__ == '__main__':
    for severity in range(1, 6):
        image_blur_plot(image_filename, severity)

