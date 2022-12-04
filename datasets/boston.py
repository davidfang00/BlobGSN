import os
import json
import math
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .dataset_utils import listdir_nohidden, normalize_trajectory, random_rotation_augment


class BostonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        seq_len=10,
        step=1,
        img_res=64,
        depth=True,
        center='first',
        normalize_rotation=True,
        rot_aug=False,
        single_sample_per_trajectory=False,
        samples_per_epoch=10000,
        **kwargs
    ):

        self.episode_len = 130
        self.data_dir = data_dir
        self.split = split
        self.datapath = os.path.join(self.data_dir, self.split)
        self.seq_len = seq_len
        self.img_res = img_res
        self.depth = depth
        self.center = center
        self.normalize_rotation = normalize_rotation
        self.seq_idxs = listdir_nohidden(self.datapath)
        self.rot_aug = rot_aug
        self.single_sample_per_trajectory = single_sample_per_trajectory
        self.samples_per_epoch = samples_per_epoch

        # step*seq_len < total_seq_len
        step = min(step, self.episode_len / seq_len)
        self.step = step
        self.resize_transform_rgb = transforms.Compose([transforms.Resize(self.img_res), transforms.CenterCrop(self.img_res), transforms.ToTensor()])
        self.resize_transform_depth = transforms.Compose([transforms.Resize(self.img_res), transforms.CenterCrop(self.img_res)])
        
        # self.resize_transform_rgb = transforms.Compose([transforms.ToTensor()])
        # self.resize_transform_depth = transforms.Compose([])
        

    def get_trajectory_Rt(self):
        Rt = []
        for idxstr in self.seq_idxs:
            episode_Rt = []
            episode_path = os.path.join(self.datapath, idxstr)
            with open(os.path.join(episode_path, 'cameras.json'), 'r') as f:
                cameras = json.load(f)

                for i in range(0, self.episode_len):
                    episode_Rt.append(torch.Tensor(cameras[i]['Rt']))
            episode_Rt = torch.stack(episode_Rt, dim=0)

            trim = episode_Rt.shape[0] % (self.seq_len * self.step)
            episode_Rt = episode_Rt[: episode_Rt.shape[0] - trim]
            Rt.append(episode_Rt)

        Rt = torch.stack(Rt, dim=0)

        # this basically samples points at the stride length
        Rt = Rt.view(-1, self.seq_len, self.step, 4, 4).permute(0, 2, 1, 3, 4).reshape(-1, self.seq_len, 4, 4)

        if self.center is not None:
            Rt = normalize_trajectory(Rt, center=self.center, normalize_rotation=self.normalize_rotation)

        if self.single_sample_per_trajectory:
            # randomly select a single point along each trajectory
            selected_indices = torch.multinomial(torch.ones(Rt.shape[:2]), num_samples=1).squeeze()
            bool_mask = torch.eye(self.seq_len)[selected_indices].bool()
            Rt = Rt[bool_mask].unsqueeze(1)

        if self.rot_aug:
            for i in range(Rt.shape[0]):
                Rt[i] = random_rotation_augment(Rt[i])
        return Rt

    def __len__(self):
        if self.samples_per_epoch:
            return self.samples_per_epoch
        else:
            trajectory_len = self.seq_len * self.step
            n_val_trajectories = int(len(self.seq_idxs) * math.floor(self.episode_len / trajectory_len))
            return n_val_trajectories

    def __getitem__(self, idx):
        random.seed()

        trajectory_len = self.seq_len * self.step

        if self.samples_per_epoch:
            idx = random.randint(0, len(self.seq_idxs) - 1)
            idxstr = self.seq_idxs[idx]
            seq_start = random.randint(0, self.episode_len - trajectory_len)
        else:
            trajectories_per_episode = math.floor(self.episode_len / trajectory_len)
            seq_idx = int(idx / trajectories_per_episode)
            seq_idx = int(self.seq_idxs[seq_idx].split("_")[-1])
            idxstr = str(seq_idx).zfill(2)

            seq_start = (idx % trajectories_per_episode) * trajectory_len
            seq_start = int(seq_start)

        # Load cameras
        episode_path = os.path.join(self.datapath, idxstr)
        with open(os.path.join(episode_path, 'cameras.json'), 'r') as f:
            cameras = json.load(f)

        Rt = []
        K = []
        rgb = []
        depth = []

        sample_indices = list(range(seq_start, seq_start + (self.seq_len * self.step), self.step))
        for idx, i in enumerate(sample_indices):
            Rt.append(torch.Tensor(cameras[i]['Rt']))
            K.append(torch.Tensor(cameras[i]['K']))
            
            _rgb = os.path.join(episode_path, str(i).zfill(3) + '_rgb.png')
            _rgb = self.resize_transform_rgb(Image.open(_rgb))
            rgb.append(_rgb[:3, :, :])

            if self.depth:
                _depth = os.path.join(episode_path, str(i).zfill(3) + '_depth.png')
                # We dont want to normalize depth values
                _depth = self.resize_transform_depth(Image.open(_depth))
                depth.append(torch.from_numpy(np.array(_depth)).unsqueeze(0))

        rgb = torch.stack(rgb)
        depth = torch.stack(depth).float()
        K = torch.stack(K)
        Rt = torch.stack(Rt)

        Rt = Rt.unsqueeze(0)  # add batch dimension
        Rt = normalize_trajectory(Rt, center=self.center, normalize_rotation=self.normalize_rotation)
        Rt = Rt[0]  # remove batch dimension

        if self.single_sample_per_trajectory:
            selected_indices = torch.multinomial(torch.ones(Rt.shape[0]), num_samples=1).squeeze()
            rgb = rgb[selected_indices].unsqueeze(0)
            depth = depth[selected_indices].unsqueeze(0)
            K = K[selected_indices].unsqueeze(0)
            Rt = Rt[selected_indices].unsqueeze(0)

        if self.rot_aug:
            Rt = random_rotation_augment(Rt)

        # Normalize K to img_res
        K = K[:, :3, :3]

        # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
        # images were rendered at 512x512 res
        # fx = 256.0 / np.tan(np.deg2rad(90.0) / 2)
        # fy = 256.0 / np.tan(np.deg2rad(90.0) / 2)
        # K[:, 0, 0] = K[:, 0, 0] * fx
        # K[:, 1, 1] = K[:, 1, 1] * fy

        downsampling_ratio = self.img_res / 114
        K[:, 0, 0] = K[:, 0, 0] * downsampling_ratio
        K[:, 1, 1] = K[:, 1, 1] * downsampling_ratio
        depth = depth # recommended scaling from game engine units to real world units

        if self.depth:
            sample = {'rgb': rgb, 'depth': depth, 'K': K, 'Rt': Rt, 'scene_idx': idx}
        else:
            sample = {'rgb': rgb, 'K': K, 'Rt': Rt, 'scene_idx': idx}

        return sample
