#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:corruption_on_test.py
@time:2024/10/30 11:09:30
@author:Yao Yongrui
'''

'''
在示例图片上测试失效条件模拟的效果
1. 模拟局部光斑
'''

import numpy as np
import cv2
import os
from Camera_corruptions import ImageAddSunMono

np.random.seed(2022)

class CorruptionBase():
    def __init__(self, corruption_severity_dict={'sun_sim':0}):
        if 'sun_sim' in corruption_severity_dict:
            self.sun_sim = ImageAddSunMono(corruption_severity_dict['sun_sim'])

    def test(self, input_filename, output_filename):
        np.random.seed(2022)
        # 读取文件夹中的全部图片路径
        pic_path_ls = os.listdir(input_filename)
        for pic_path in pic_path_ls:
            pic_path = os.path.join(input_filename, pic_path)
            img_bgr_255_np_uint8 = cv2.imread(pic_path)
            img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
            image_aug_rgb = self.sun_sim(
                image=img_rgb_255_np_uint8
            )
            image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(output_filename, pic_path.split('/')[-1]), image_aug_bgr)
        #img_bgr_255_np_uint8 = cv2.imread(input_filename)
    
    def plot(self, pic_path, num, output_filename):

        if not os.path.exists(output_filename):
            os.makedirs(output_filename)
        
        np.random.seed(2022)
        img_bgr_255_np_uint8 = cv2.imread(pic_path)
        img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
        image_aug_rgb = self.sun_sim(
            image=img_rgb_255_np_uint8
        )
        image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
        output_path = os.path.join(output_filename, f'0{num}.jpg')
        cv2.imwrite(output_path, image_aug_bgr)

if __name__ == '__main__':
    input_filename = './nuscenes_example_pic'
    output_filename = './corruption_example_pic'
    pic_path = './nuscenes_example_pic/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg'
    plot_dirname = './local_spot_plot'

    # corruption_severity_dict = {'sun_sim':80}
    # corruption = CorruptionBase(corruption_severity_dict)
    # corruption(input_filename, output_filename)
    # print('Corruption done!')

    severity_ls = [20, 40, 60, 80]
    for i in range(len(severity_ls)):
        corruption_severity_dict = {'sun_sim':severity_ls[i]}
        corruption = CorruptionBase(corruption_severity_dict)
        corruption.plot(pic_path, i + 1, plot_dirname)
    print('Plot done!')