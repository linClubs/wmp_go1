# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from typing import Tuple

import datetime
import json
import os
import subprocess

import torch
import numpy as np

def git_store_code_state(log_dir, repo_dir=None):
    if log_dir is None:
        return

    repo_dir = repo_dir or os.getcwd()

    def run_git(args):
        return subprocess.run(
            ["git"] + args,
            cwd=repo_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        ).stdout

    try:
        repo_dir = run_git(["rev-parse", "--show-toplevel"]).strip()
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        head = run_git(["rev-parse", "HEAD"]).strip()
        status = run_git(["status", "--short", "--branch"])
        diff = run_git(["diff", "--binary", "HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[WARNING] Failed to store git code state: {e}")
        return

    git_dir = os.path.join(log_dir, "git")
    os.makedirs(git_dir, exist_ok=True)

    meta = {
        "repo_dir": repo_dir,
        "branch": branch,
        "head": head,
        "is_dirty": bool(status.strip().splitlines()[1:]),
        "created_at": datetime.datetime.now().astimezone().isoformat(),
    }

    with open(os.path.join(git_dir, "git_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    with open(os.path.join(git_dir, "git_status.txt"), "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(status)
    with open(os.path.join(git_dir, "git_diff.patch"), "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(diff)
    restore_path = os.path.join(git_dir, "restore_code.sh")
    with open(restore_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
        f.write('cd "$(git rev-parse --show-toplevel)"\n')
        f.write(f"git checkout {head}\n")
        f.write('git apply "$SCRIPT_DIR/git_diff.patch"\n')
    os.chmod(restore_path, 0o755)

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip(
            (input - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(
            self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(
            self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp(
            (input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
                expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(
                torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = torch.nn.functional.normalize

    def forward(self, x):
        x = self.normalize(x, dim=-1)
        return x


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """对 [x, y, z, w] 四元数做球面插值，支持单个和批量输入且不修改输入。

    fraction 可为标量、批量形状或批量形状加末尾单维。
    非最短路径下，互为相反数的四元数沿用起始姿态。
    """
    q0, q1 = torch.broadcast_tensors(q0, q1)
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    fraction = torch.as_tensor(fraction, dtype=q0.dtype, device=q0.device)
    if fraction.ndim == q0.ndim - 1:
        fraction = fraction.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1., 1.)
    if shortestpath:
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()
    # 重合端点改用线性插值，避免 acos 端点和除零引起数值问题。
    eps = torch.finfo(q0.dtype).eps * 4
    coincident = dot.abs() >= 1 - eps
    angle = torch.acos(dot.clamp(-1 + eps, 1 - eps)) + spin * torch.pi
    sine = torch.sin(angle)
    safe_sine = torch.where(sine.abs() < eps, torch.ones_like(sine), sine)
    spherical = (torch.sin((1 - fraction) * angle) * q0 + torch.sin(fraction * angle) * q1) / safe_sine
    linear = torch.nn.functional.normalize((1 - fraction) * q0 + fraction * q1, dim=-1)
    fallback = torch.where(dot < 0, q0, linear)
    result = torch.where(coincident | (sine.abs() < eps), fallback, spherical)
    result = torch.where(fraction == 0, q0, result)
    return torch.where(fraction == 1, q1, result)
