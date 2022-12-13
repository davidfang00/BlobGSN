import os
import random
import sys
import time
import urllib.request

import imageio as imio
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from IPython.display import HTML, clear_output, display
from PIL import Image

sys.path.append('..')

from builders.builders import build_dataloader
from models.model_utils import TrajectorySampler
from notebooks.walkthrough_utils import get_smooth_trajectory
from utils.camera_trajectory import go_backward, go_forward, go_upward, rotate_n
from utils.utils import instantiate_from_config

torch.manual_seed(10000)

checkpoint_filename = 'logsReplica/checkpoints/last.ckpt'
data_path = 'data/replica_all'

checkpoint = torch.load(checkpoint_filename)
state_dict = checkpoint['state_dict']

print("checkpoint loaded")

# get rid of all the inception params which are leftover from FID metric
keys_for_deletion = []
for key in state_dict.keys():
    if 'fid' in key:
        keys_for_deletion.append(key)
for key in keys_for_deletion:
    del state_dict[key]

opt = checkpoint['opt']
opt.data_config.data_dir = data_path

# create dataloader
data_module = build_dataloader(opt.data_config, verbose=False)

# collect a set of real trajectories from the dataset
real_Rts = data_module.train_loader.dataset.get_trajectory_Rt()
trajectory_sampler = TrajectorySampler(real_Rts=real_Rts, mode=opt.model_config.params.trajectory_mode)

# initialize model and load state
gsn = instantiate_from_config(opt.model_config).cuda().eval()
gsn.set_trajectory_sampler(trajectory_sampler=trajectory_sampler)
gsn.load_state_dict(state_dict, strict=True)

# increase nerf_out_res after loading model to generate higher res samples (i.e., implicit upsampling)
gsn.generator_config.params.nerf_out_res *= 2
# trade render speed for memory by rendering smaller patches at a time
gsn.patch_size = 32

# load a batch of data so that we can use the camera parameters
real_data = next(iter(data_module.train_dataloader()))
print('model initialized and loaded')

device = 'cuda'

def sample(id, save_dir):
    # sample z noise latent
    z_dim = opt.model_config.params.blob_maker_config.params.noise_dim
    z = torch.randn(1, z_dim, device=device)

    # plot blobs
    with torch.no_grad():
        layout = gsn.blob_maker(z, viz = True, viz_size=[256, 256])
    def for_canvas(img):
        return img.detach()[0].round().permute(1, 2, 0).clamp(min=0, max=255).cpu().numpy().astype(np.uint8)
    blob_im = layout['feature_img']
    blobs = for_canvas(blob_im.mul(255))
    print(blobs.shape)

    # intrinsic camera matrix from real data
    K_current = real_data['K'].to(device)
    # initialize extrinsic camera matrix at the center of the scene
    Rt_current = torch.eye(4, device=device).view(1, 1, 4, 4)

    # get trajectory
    trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
    actions = ['d'] * 30 + ['w'] * 2 + ['s'] * 2
    actions.append('q')
    for action in actions:
        camera_params = {'K': K_current, 'Rt': Rt_current}
        
        with torch.no_grad():
            fake_rgb, fake_depth, Rt, K = gsn.generate_from_layout(layout, camera_params=camera_params) 
        
        trajectory['rgb'].append(fake_rgb)
        trajectory['depth'].append(fake_depth)
        trajectory['Rt'].append(Rt)
        trajectory['K'].append(K)
        
        rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t h) w c').cpu()
        
        clear_output()
        fig = plt.figure(figsize = (8, 8)) 
        ax = fig.add_subplot(111)
        ax.imshow(rgb_current, interpolation='bilinear')
        ax.set_title('Current observation');
        plt.axis('off')
        plt.show()

        step_size = opt.model_config.params.voxel_size / 0.6

        if action == 'a':
            # Turn left
            Rt = rotate_n(n=-30.0).to(device)
            Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
        if action == 'd':
            # Turn right
            Rt = rotate_n(n=30.0).to(device)
            Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
        if action == 'w':
            # Go forward
            Rt_current = go_forward(Rt_current, step=step_size)
        if action == 's':
            # Go backward
            Rt_current = go_backward(Rt_current, step=step_size)

        if action == 'u':
            Rt_current = go_upward(Rt_current, step=step_size)
            
        if action == 'q':
            break
            
    for key in trajectory.keys():
        trajectory[key] = torch.cat(trajectory[key], dim=1)
        
    print('trajectory finished')
        
    # jitter camera pose a tiny amount to make sure each pose is unique
    # (to avoid problems with trajectory smoothing)
    trajectory['Rt'] = trajectory['Rt'] + torch.rand_like(trajectory['Rt']) * 1e-5

    # fit a smooth spline to the trajectory keypoints
    # n_keypoints = len(trajectory['Rt'][0])
    # new_Rts = trajectory['Rt'][0].unsqueeze(1)
    # n_steps = len(new_Rts)
    # print(trajectory['Rt'][0].shape)
    # print(new_Rts.shape)

    n_keypoints = len(trajectory['Rt'][0])
    new_Rts = get_smooth_trajectory(Rt=trajectory['Rt'][0], n_frames=5 * n_keypoints, subsample=3)
    n_steps = len(new_Rts)

    # render frames along new smooth trajectory
    resampled_trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
    for i in range(n_steps):
        if i % 10 == 0:
            print('Rendering frame {}/{}'.format(i, n_steps))
        
        camera_params = {'K': K_current[:1, :1], 'Rt': new_Rts[i:i+1].to(device)}
        
        with torch.no_grad():
            fake_rgb, fake_depth, Rt, K = gsn.generate_from_layout(layout, camera_params=camera_params)

        resampled_trajectory['rgb'].append(fake_rgb)
        resampled_trajectory['depth'].append(fake_depth)
        resampled_trajectory['Rt'].append(Rt)
        resampled_trajectory['K'].append(K)

    for key in resampled_trajectory.keys():
        resampled_trajectory[key] = torch.cat(resampled_trajectory[key], dim=1)

    imgs = []
    for i in range(n_steps):
        im = resampled_trajectory['rgb'][0, i].permute(1, 2, 0).detach().cpu()
        imgs.append((im * 255).byte())

    # save gif with random unique name
    # n = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
    n = id
    animation_filename = f'{save_dir}/{n}_camera_trajectory.gif'
    imio.mimsave(animation_filename, imgs, fps=30)
    print('Saving animation to {}\n'.format(animation_filename))

    blob_filename = f'{save_dir}/{n}_blobs.png'
    im = Image.fromarray(blobs)
    im.save(blob_filename)
    print('Blob image saved to {}\n'.format(blob_filename))

if not os.path.exists('walkthroughs'):
    os.mkdir('walkthroughs')

dir = "walkthroughs/{}_walkthrough".format(round(time.time() * 1000))
os.mkdir(dir)

for i in range(3):
    sample(i, dir)