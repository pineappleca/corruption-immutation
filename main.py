#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:main.py
@time:2024/09/13 21:33:07
@author:Yao Yongrui
'''

# weather-level
from .Camera_corruptions import ImageAddSnow,ImageAddFog,ImageAddRain

severity = 1
snow_sim = ImageAddSnow(severity, seed=2022)


img_bgr_255_np_uint8 = results['img'] # the img in mmdet3d loading pipeline
img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
image_aug_rgb = snow_sim(
    image=img_rgb_255_np_uint8
    )
image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
results['img'] = image_aug_bgr


# other corruption can be the same operation, like ImageAddGaussianNoise,ImageAddImpulseNoise,ImageAddUniformNoise

# Note that the object-level corruption need 3D bounding box information as input, like results['gt_bboxes_3d'] in mmdet3d
from .Camera_corruptions import ImageBBoxOperation
bbox_shear = ImageBBoxOperation(severity)
img_bgr_255_np_uint8 = results['img']
bboxes_corners = results['gt_bboxes_3d'].corners
bboxes_centers = results['gt_bboxes_3d'].center
lidar2img = results['lidar2img'] 
img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
c = [0.05, 0.1, 0.15, 0.2, 0.25][bbox_shear.severity - 1]
b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
transform_matrix = torch.tensor([
    [1, 0, b],
    [d, 1, e],
    [f, 0, 1]
]).float()

image_aug_rgb = bbox_shear(
    image=img_rgb_255_np_uint8,
    bboxes_centers=bboxes_centers,
    bboxes_corners=bboxes_corners,
    transform_matrix=transform_matrix,
    lidar2img=lidar2img,
)
image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
results['img'] = image_aug_bgr