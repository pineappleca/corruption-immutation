#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:目标模糊_.py
@time:2024/11/16 21:01:39
@author:Yao Yongrui
'''

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix, view_points, BoxVisibility
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from Camera_corruptions import ImageBBoxMotionBlurFrontBackMono
# from mmdet3d.core.bbox.structures.utils import points_cam2img
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt

# 初始化 Nuscenes 数据集
nusc = NuScenes(version='v1.0-mini', dataroot='../BEVFormer/data/nuscenes', verbose=True)
# image_filename = 'samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
cam = 'CAM_FRONT'
image_filename = 'samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460.jpg'
out_dir = './target_blur_plot'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def annotation3d22d(image_filename, cam):
    # 获取 sample 数据
    # 查询路径为 'filename' -> 'sample_data' -> 'sample' -> 'anns'
    sample_data = nusc.get('sample_data', nusc.field2token('sample_data', 'filename', image_filename)[0])
    sample = nusc.get('sample', sample_data['sample_token'])
    annotations = sample['anns']

    annotations_2d_ls = []

    # 设置可见性级别
    box_vis_level = BoxVisibility.ANY

    # for ann_token in annotations:
    #     ann_record = nusc.get('sample_annotation', ann_token)
    #     # sample_record = nusc.get('sample', ann_record['sample_token'])

    # 获取目标的3D bounding box
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample['data'][cam], box_vis_level=box_vis_level,
                                                    selected_anntokens=annotations)
    for box in boxes: 
        corners_2d = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
        array_corners_2d = np.array(corners_2d, np.int32)
        array_corners_reshaped = array_corners_2d.T.reshape(8, 1, 2)
        annotations_2d_ls.append(array_corners_reshaped)
    return annotations_2d_ls  

def motion_target_blur(image_filename, output_dir, gt_corners_ls, severity):

    corruption = 0.02 * (severity - 1)

    image = cv2.imread(os.path.join(nusc.dataroot, image_filename))
    canvas = image.copy()
    mask = np.zeros((canvas.shape[0],canvas.shape[1]))
    for corners in gt_corners_ls:
        mask = cv2.fillConvexPoly(mask, corners, 1)
    mask_bool_float = (mask>0).astype(np.float32)[:,:,None]
    ibmb = ImageBBoxMotionBlurFrontBackMono(severity)
    image_aug_layer = ibmb.zoom_blur(image, corruption)
    images_aug = image_aug_layer * mask_bool_float + (1-mask_bool_float) * image
    image_aug = images_aug.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f'0{severity}.jpg'), image_aug)

if __name__ == '__main__':
    annotations_2d_ls = annotation3d22d(image_filename, cam)
    for i in range(1, 6):
        motion_target_blur(image_filename, out_dir, annotations_2d_ls, i)


