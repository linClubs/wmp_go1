# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026-2027 zh
"""内容：使用 Torch 处理动作片段的姿态标准化与航向。"""
import torch

from rsl_rl.datasets import pose3d


def standardize_quaternion(q):
    """将单个或批量四元数统一为 w >= 0，不修改输入。"""
    q = pose3d.as_float_tensor(q)
    return torch.where(q[..., 3:] < 0, -q, q)


def normalize_rotation_angle(theta):
    """将弧度角归一化到 [-π, π]，保留原有正负端点。"""
    theta = pose3d.as_float_tensor(theta)
    wrapped = torch.fmod(theta, 2 * torch.pi)
    wrapped = torch.where(wrapped >= 0, wrapped - 2 * torch.pi, wrapped + 2 * torch.pi)
    return torch.where(torch.abs(theta) > torch.pi, wrapped, theta)


def calc_heading(q):
    """计算旋转后的 x 轴在水平面上的航向角，支持批量输入。"""
    rot_dir = pose3d.QuaternionRotatePoint([1., 0., 0.], q)
    return torch.atan2(rot_dir[..., 1], rot_dir[..., 0])


def calc_heading_rot(q):
    """计算绕 z 轴的航向四元数。"""
    heading = calc_heading(q)
    axis = heading.new_tensor([0., 0., 1.])
    return pose3d.QuaternionFromAxisAngle(axis, heading)
