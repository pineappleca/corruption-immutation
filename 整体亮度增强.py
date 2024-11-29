#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:整体亮度增强.py
@time:2024/10/30 15:26:36
@author:Yao Yongrui
'''

'''
在示例图片上测试失效条件模拟的效果
1. 模拟整体光照增强
'''

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('nuscenes_example_pic/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg')
# img = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)
severity_ls = [0, 20, 40, 60, 80]
aug_output_filename = './light_aug_plot'
des_output_filename = './light_des_plot'

if not os.path.exists(aug_output_filename):
    os.makedirs(aug_output_filename)

if not os.path.exists(des_output_filename):
    os.makedirs(des_output_filename)

# 整体亮度增强
# for i in range(len(severity_ls)):
#     beta = 20 + severity_ls[i]
#     result = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
#     result_uint8 = result.copy().astype(np.uint8)
#     print(result)
#     cv2.imwrite(os.path.join(output_filename, f'0{i + 1}.jpg'), result)
for i in range(len(severity_ls)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 50 + 0.5 * severity_ls[i])
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(aug_output_filename, f'0{i + 1}.jpg'), result)

# # 整体亮度减弱
# for i in range(len(severity_ls)):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     v = cv2.add(v, - 60 - 0.5 * severity_ls[i])
#     v = np.clip(v, 0, 255)
#     final_hsv = cv2.merge((h, s, v))
#     result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     cv2.imwrite(os.path.join(des_output_filename, f'0{i + 1}.jpg'), result)

# #获取图像行和列
# rows, cols = img.shape[0], img.shape[1]
 
# #设置中心点
# centerX = rows / 2
# centerY = cols / 2
# print (centerX, centerY)
# radius = min(centerX, centerY)
# print (radius)
 
# #设置光照强度
# strength = 200
 
# #图像光照特效
# for i in range(rows):
#     for j in range(cols):
#         #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
#         distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
#         #获取原始图像
#         B = img[i,j][0]
#         G = img[i,j][1]
#         R = img[i,j][2]
#         if (distance < pow(radius, 2)):
#             #按照距离大小计算增强的光照值
#             result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
#             B = img[i,j][0] + result
#             G = img[i,j][1] + result
#             R = img[i,j][2] + result
#             #判断边界 防止越界
#             B = min(255, max(0, B))
#             G = min(255, max(0, G))
#             R = min(255, max(0, R))
#             img[i,j] = np.uint8((B, G, R))
#         else:
#             img[i,j] = np.uint8((B, G, R))
        
# #显示图像
# cv2.imwrite('./corruption_example_pic/test.jpg', img)
# # plt.imshow(img)
# # plt.show()

# from Camera_corruptions import ImageLightAug
# import numpy as np
# import cv2
# import os

# np.random.seed(2022)

# class CorruptionBase():
#     def __init__(self, corruption_severity_dict={'light_aug':20}):
#         if 'light_aug' in corruption_severity_dict:
#             self.light_aug = ImageLightAug(corruption_severity_dict['light_aug'])

#     def test(self, input_filename, output_filename):
#         np.random.seed(2022)
#         # 读取文件夹中的全部图片路径
#         pic_path_ls = os.listdir(input_filename)
#         for pic_path in pic_path_ls:
#             pic_path = os.path.join(input_filename, pic_path)
#             img_bgr_255_np_uint8 = cv2.imread(pic_path)
#             img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
#             image_aug_rgb = self.light_aug(
#                 image=img_rgb_255_np_uint8
#             )
#             image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
#             cv2.imwrite(os.path.join(output_filename, pic_path.split('/')[-1]), image_aug_bgr)
#         #img_bgr_255_np_uint8 = cv2.imread(input_filename)
    
#     def plot(self, pic_path, num, output_filename):

#         if not os.path.exists(output_filename):
#             os.makedirs(output_filename)
        
#         np.random.seed(2022)
#         img_bgr_255_np_uint8 = cv2.imread(pic_path)
#         img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
#         image_aug_rgb = self.light_aug(
#             image=img_rgb_255_np_uint8
#         )
#         image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
#         output_path = os.path.join(output_filename, f'0{num}.jpg')
#         cv2.imwrite(output_path, image_aug_bgr)

# if __name__ == '__main__':
#     input_filename = './nuscenes_example_pic'
#     output_filename = './corruption_example_pic'
#     pic_path = './nuscenes_example_pic/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg'
#     plot_dirname = './light_aug_plot'

#     corruption_severity_dict = {'light_aug':20}
#     corruption = CorruptionBase(corruption_severity_dict)
#     corruption.test(input_filename, output_filename)
#     print('Corruption done!')

    # severity_ls = [20, 40, 60, 80]
    # for i in range(len(severity_ls)):
    #     corruption_severity_dict = {'sun_sim':severity_ls[i]}
    #     corruption = CorruptionBase(corruption_severity_dict)
    #     corruption.plot(pic_path, i + 1, plot_dirname)
    # print('Plot done!')