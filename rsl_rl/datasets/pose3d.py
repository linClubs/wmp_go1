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
"""内容：使用 Torch 处理三维向量和姿态，四元数顺序为 [x, y, z, w]。"""
import math

import torch


VECTOR3_0 = torch.zeros(3, dtype=torch.float64)
VECTOR3_1 = torch.ones(3, dtype=torch.float64)
VECTOR3_X = torch.tensor([1., 0., 0.], dtype=torch.float64)
VECTOR3_Y = torch.tensor([0., 1., 0.], dtype=torch.float64)
VECTOR3_Z = torch.tensor([0., 0., 1.], dtype=torch.float64)
QUATERNION_IDENTITY = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)


def as_float_tensor(value):
    """保留浮点 Tensor 的类型及设备，其他输入转换为浮点 Tensor。"""
    value = torch.as_tensor(value)
    return value if value.is_floating_point() else value.to(torch.get_default_dtype())


def Vector3RandomNormal(sigma, mu=VECTOR3_0):
    """生成各分量独立的三维正态随机向量。"""
    mu = as_float_tensor(mu)
    return torch.randn_like(mu) * torch.as_tensor(sigma, device=mu.device, dtype=mu.dtype) + mu


def Vector3RandomUniform(low=VECTOR3_0, high=VECTOR3_1):
    """在给定边界内生成三维均匀随机向量。"""
    low = as_float_tensor(low)
    high = torch.as_tensor(high, dtype=low.dtype, device=low.device)
    return low + torch.rand_like(low) * (high - low)


def Vector3RandomUnit():
    """在单位球面上均匀生成三维单位向量。"""
    vector = torch.randn(3, dtype=torch.float64)
    return vector / torch.linalg.vector_norm(vector)


def QuaternionNormalize(q):
    """归一化单个或批量四元数，拒绝零长度输入。"""
    q = as_float_tensor(q)
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if torch.any(torch.isclose(norm, torch.zeros_like(norm))):
        raise ValueError("Quaternion must have nonzero length")
    return q / norm


def QuaternionFromAxisAngle(axis, angle):
    """将旋转轴和弧度角转换为单位四元数，支持批量广播。"""
    axis = as_float_tensor(axis)
    if axis.shape[-1] != 3:
        raise ValueError("Axis must have three components")
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if torch.any(torch.isclose(norm, torch.zeros_like(norm))):
        raise ValueError("Axis must have nonzero length")
    half_angle = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device) * 0.5
    xyz, w = torch.broadcast_tensors(axis / norm * torch.sin(half_angle)[..., None], torch.cos(half_angle)[..., None])
    return torch.cat((xyz, w[..., :1]), dim=-1)


def QuaternionToAxisAngle(quat, default_axis=VECTOR3_Z, direction_axis=None):
    """将单位四元数转换为旋转轴与弧度角，支持批量输入。"""
    quat = as_float_tensor(quat)
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion must have four components")
    norm = torch.linalg.vector_norm(quat, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm)):
        raise ValueError("Quaternion must have unit length")
    default_axis = torch.as_tensor(default_axis, dtype=quat.dtype, device=quat.device)
    if default_axis.shape[-1] != 3 or not torch.allclose(
        torch.linalg.vector_norm(default_axis, dim=-1), torch.ones_like(default_axis[..., 0])
    ):
        raise ValueError("Default axis must be a unit vector")
    axis_norm = torch.linalg.vector_norm(quat[..., :3], dim=-1, keepdim=True)
    axis = torch.where(axis_norm < 1e-8, default_axis, quat[..., :3] / axis_norm.clamp_min(1e-8))
    sin_half_angle = axis_norm[..., 0]
    if direction_axis is not None:
        direction_axis = torch.as_tensor(direction_axis, dtype=quat.dtype, device=quat.device)
        flip = (axis * direction_axis).sum(dim=-1) < 0
        axis = torch.where(flip[..., None], -axis, axis)
        sin_half_angle = torch.where(flip, -sin_half_angle, sin_half_angle)
    return axis, 2 * torch.atan2(sin_half_angle, quat[..., 3])


def QuaternionRandomRotation(max_angle=math.pi):
    """围绕随机单位轴生成角度位于给定范围的旋转。"""
    return QuaternionFromAxisAngle(Vector3RandomUnit(), torch.rand((), dtype=torch.float64) * max_angle)


def QuaternionMultiply(q0, q1):
    """计算 Hamilton 四元数乘积，支持批量广播。"""
    q0 = as_float_tensor(q0)
    q1 = torch.as_tensor(q1, dtype=q0.dtype, device=q0.device)
    q0, q1 = torch.broadcast_tensors(q0, q1)
    v0, v1 = q0[..., :3], q1[..., :3]
    xyz = q0[..., 3:] * v1 + q1[..., 3:] * v0 + torch.cross(v0, v1, dim=-1)
    w = q0[..., 3:] * q1[..., 3:] - (v0 * v1).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def QuaternionInverse(quat):
    """计算非零四元数的逆，支持非单位四元数。"""
    quat = as_float_tensor(quat)
    norm_squared = (quat * quat).sum(dim=-1, keepdim=True)
    if torch.any(norm_squared == 0):
        raise ValueError("Quaternion must have nonzero length")
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1) / norm_squared


def QuaternionRotatePoint(point, quat):
    """使用 q * p * q逆旋转三维向量，支持批量广播。"""
    quat = as_float_tensor(quat)
    point = torch.as_tensor(point, dtype=quat.dtype, device=quat.device)
    q_point = torch.cat((point, torch.zeros_like(point[..., :1])), dim=-1)
    return QuaternionMultiply(QuaternionMultiply(quat, q_point), QuaternionInverse(quat))[..., :3]


def IsRotationMatrix(m):
    """检查矩阵左上角的 3×3 部分是否正交，保持原有判定语义。"""
    m = as_float_tensor(m)
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 3:
        raise ValueError("Matrix must be at least 3x3")
    rot = m[:3, :3]
    return bool(torch.allclose(rot @ rot.T, torch.eye(3, dtype=m.dtype, device=m.device), atol=1e-4))
