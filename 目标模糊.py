#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:目标模糊.py
@time:2024/11/15 15:36:43
@author:Yao Yongrui
'''

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
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
# image_filename = 'samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460.jpg'
image_filename = 'samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg'

# def points_cam2img(points_3d, cam_intrinsic):
#     """
#     将 3D 点从相机坐标系转换到图像坐标系，测试Camera_corruptions库中的函数

#     参数:
#     points_3d (numpy.ndarray): 形状为 (N, 3) 的 3D 点数组。
#     cam_intrinsic (numpy.ndarray): 形状为 (3, 3) 的相机内参矩阵。

#     返回:
#     numpy.ndarray: 形状为 (N, 2) 的 2D 图像坐标数组。
#     """
#     # 将 3D 点转换为齐次坐标
#     points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
#     # 使用相机内参矩阵将 3D 点投影到图像平面
#     points_2d_hom = np.dot(cam_intrinsic, points_3d_hom.T).T
    
#     # 转换为非齐次坐标
#     points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2, np.newaxis]
    
#     return points_2d

def transform_points(points, transform_matrix):
    """
    使用变换矩阵将点从一个坐标系转换到另一个坐标系。

    参数:
    points (numpy.ndarray): 形状为 (N, 3) 的点数组。
    transform_matrix (numpy.ndarray): 形状为 (4, 4) 的变换矩阵。

    返回:
    numpy.ndarray: 形状为 (N, 3) 的转换后的点数组。
    """
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_hom = np.dot(transform_matrix, points_hom.T).T
    return points_transformed_hom[:, :3]


def get_annotations_from_image(image_filename):
    # 获取 sample 数据
    # 查询路径为 'filename' -> 'sample_data' -> 'sample' -> 'anns'
    sample_data = nusc.get('sample_data', nusc.field2token('sample_data', 'filename', image_filename)[0])
    # print(sample_data)
    sample = nusc.get('sample', sample_data['sample_token'])
    # print(sample)
    annotations = sample['anns']
    # print([[i, nusc.get('sample_an notation', annotations[i])] for i in range(len(annotations))])
    
    # 获取 3D 检测框
    count = 0
    gt_bboxes_3d = []

    # 尝试直接使用box的角点坐标
    nusc_bboxes_corners = []
    for ann_token in annotations:
        ann = nusc.get('sample_annotation', ann_token)
        # print(count)
        # print(ann)
        # if count == 14:
        # print(ann)
        nusc.render_annotation(ann_token)
        # plt.savefig(f'annotation_render{count}.png')
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation'])
        )
        # center = box.center
        # size = box.wlh
        # orientation = box.orientation
        # yaw = orientation.yaw_pitch_roll[0]
        # gt_bboxes_3d.append([center[0], center[1], center[2], size[0], size[1], size[2], yaw])
        # nusc_bboxes_corners.append(box.corners().T)
        # count += 1
        
        # 将 3D bounding box 的角点从车辆坐标系转换到相机坐标系
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        vehicle_to_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        global_to_sensor = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=True)
        vehicle_to_sensor = np.dot(global_to_sensor, vehicle_to_global)
        corners_3d_vehicle = box.corners().T
        corners_3d_sensor = transform_points(corners_3d_vehicle, vehicle_to_sensor)

        gt_bboxes_3d.append(corners_3d_sensor)

    
    gt_bboxes_3d = np.array(gt_bboxes_3d)
    # gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
    
    

    # 获取相机内参
    calibrated_sensor_token = sample_data['calibrated_sensor_token']
    calibrated_sensor = nusc.get('calibrated_sensor', calibrated_sensor_token)
    print(calibrated_sensor)
    cam_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
    cam_intrinsic = torch.tensor(cam_intrinsic).float()

    nusc_bboxes_corners = torch.tensor(nusc_bboxes_corners).float()

    # return cam_intrinsic, nusc_bboxes_corners, gt_bboxes_3d.corners, gt_bboxes_3d.center
    return cam_intrinsic, gt_bboxes_3d

def bbox_blur_aug(img_filename, cam2img, bboxes_corners, bboxes_centers, severity):
    img_filename = os.path.join('../BEVFormer/data/nuscenes', img_filename)
    bbox_blur = ImageBBoxMotionBlurFrontBackMono(severity)
    image_bgr = cv2.imread(img_filename)
    # 将 image 转为 RGB 格式
    image_rgb = image_bgr[:, :, [2, 1, 0]]
    image_aug_rgb = bbox_blur(
        image=image_rgb,
        bboxes_corners=bboxes_corners,
        bboxes_centers=bboxes_centers,
        cam2img=cam2img
    )
    return image_aug_rgb

def points_cam2img(points_3d, cam_intrinsic):
    """
    将 3D 点从相机坐标系转换到图像坐标系。

    参数:
    points_3d (numpy.ndarray): 形状为 (N, 3) 的 3D 点数组。
    cam_intrinsic (numpy.ndarray): 形状为 (3, 3) 的相机内参矩阵。

    返回:
    numpy.ndarray: 形状为 (N, 2) 的 2D 图像坐标数组。
    """
    # 将 3D 点转换为齐次坐标
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # 使用相机内参矩阵将 3D 点投影到图像平面
    points_2d_hom = np.dot(cam_intrinsic, points_3d_hom.T).T
    
    # 转换为非齐次坐标
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2, np.newaxis]
    
    return points_2d

cam2img, gt_bboxes_3d= get_annotations_from_image(image_filename)
# print(cam2img)
print(gt_bboxes_3d[9])
# print(bboxes_corners[9])
print(points_cam2img(gt_bboxes_3d[9], cam2img))
# print(points_cam2img(bboxes_corners[9], cam2img, with_depth=True))
# severity_lst = [5]
# output_dir = './target_blur_plot'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# for severity in severity_lst:
#     image_aug_rgb = bbox_blur_aug(image_filename, cam2img, bboxes_corners, bboxes_centers, severity)
#     image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]




    # output_filename = os.path.join(output_dir, f'0{severity}.jpg')
    # cv2.imwrite(output_filename, image_aug_bgr)