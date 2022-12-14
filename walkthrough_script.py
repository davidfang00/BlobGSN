import torch
import random
import urllib.request
import imageio as imio
from einops import rearrange
import matplotlib.pyplot as plt
from IPython.display import clear_output, HTML, display
import numpy as np

import sys
sys.path.append('..')

from models.model_utils import TrajectorySampler
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config
from notebooks.walkthrough_utils import get_smooth_trajectory
from utils.camera_trajectory import rotate_n, go_forward, go_backward

import copy
from PIL import ImageDraw, Image, ImageFont
import torchvision.transforms.functional as FF

COLORS = torch.tensor([[0.9804, 0.9451, 0.9176],
        [0.8980, 0.5255, 0.0235],
        [0.3647, 0.4118, 0.6941],
        [0.3216, 0.7373, 0.6392],
        [0.6000, 0.7882, 0.2706],
        [0.1843, 0.5412, 0.7686],
        [0.6471, 0.6667, 0.6000],
        [0.8549, 0.6471, 0.1059],
        [0.4627, 0.3059, 0.6235],
        [0.8000, 0.3804, 0.6902],
        [0.9294, 0.3922, 0.3529],
        [0.1412, 0.4745, 0.4235],
        [0.4000, 0.7725, 0.8000],
        [0.9647, 0.8118, 0.4431],
        [0.9725, 0.6118, 0.4549],
        [0.8627, 0.6902, 0.9490],
        [0.5294, 0.7725, 0.3725],
        [0.6196, 0.7255, 0.9529],
        [0.9961, 0.5333, 0.6941],
        [0.7882, 0.8588, 0.4549],
        [0.5451, 0.8784, 0.6431],
        [0.7059, 0.5922, 0.9059],
        [0.7020, 0.7020, 0.7020],
        [0.5216, 0.3608, 0.4588],
        [0.8510, 0.6863, 0.4196],
        [0.6863, 0.3922, 0.3451],
        [0.4510, 0.4353, 0.2980],
        [0.3216, 0.4157, 0.5137],
        [0.3843, 0.3255, 0.4667],
        [0.4078, 0.5216, 0.3608],
        [0.6118, 0.6118, 0.3686],
        [0.6275, 0.3804, 0.4667],
        [0.5490, 0.4706, 0.3647],
        [0.4863, 0.4863, 0.4863]])

checkpoint_filename = 'logsReplica/checkpoints/gsn-model-best-fid.ckpt'
data_path = 'data/replica_all'
print(f'using {checkpoint_filename}')

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

torch.seed()
device = 'cuda'
# z_dim = opt.model_config.params.decoder_config.params.z_dim
z_dim = opt.model_config.params.blob_maker_config.params.noise_dim
z = torch.randn(1, z_dim, device=device)

# plot blobs
truncate = 0.
gsn.blob_maker.device = device
gsn.blob_maker.get_mean_latent(100)

def draw_labels(img, layout, T, colors, layout_i=0):
    font = ImageFont.truetype('notebooks/LiberationSans-Bold.ttf', 10)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    mask = layout['sizes'][layout_i, 1:] > T
    idmap = torch.arange(len(mask))[mask]
    blob = {k: layout[k][layout_i][mask].mul(128).tolist() for k in ('xs', 'ys')}
    for i, (x, y) in enumerate(zip(blob['xs'], blob['ys'])):
        I = idmap[i]
        _, h = draw.textsize(str(I), font=font)
        w = h
        
        color = tuple(colors[I + 1].mul(255).round().int().tolist())
        draw.text((x - w / 2, y - h / 2), f'{I}', fill=color, stroke_width=1, font=font, stroke_fill=(0, 0, 0))
    return FF.to_tensor(img).permute(1, 2, 0), img


def for_canvas(img):
    return img.detach()[0].round().permute(1, 2, 0).clamp(min=0, max=255).cpu().numpy().astype(np.uint8)



# intrinsic camera matrix from real data
K_current = real_data['K'].to(device)
# initialize extrinsic camera matrix at the center of the scene
Rt_current = torch.eye(4, device=device).view(1, 1, 4, 4)
camera_params = {'K': K_current, 'Rt': Rt_current}

stacked_imgs = []
for i in range(10):
    torch.manual_seed(1)
    with torch.no_grad():
        
        layout = gsn.blob_maker.generate_layout(z, viz = True, viz_size=[128, 128], no_jitter = True, 
                                                ret_layout = True)

        layout_copy = copy.deepcopy(layout)
        
        # BLOB EDITING
#         layout_copy['xs'][0, 3] += i*0.3/10
        layout_copy['xs'][0, 0] += i*0.5/10

#         layout_copy['sizes'][0, 7] -= i*5/10

        del layout_copy['feature_img']
        edited_layout = gsn.blob_maker.generate_layout(layout = layout_copy, viz = True, ret_layout = True, no_jitter = True,
                                                      viz_size = [128, 128], covs_raw = False)
        
    edited_blob_im = edited_layout['feature_img']
    edited_blobs = for_canvas(edited_blob_im.mul(255))
    edited_labeled_blobs, edited_labeled_blobs_img = draw_labels(edited_blobs, edited_layout, truncate, COLORS)
        
    l = layout
    SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
    gen_input = {
            'input': l['feature_grid'],
            'styles': {k: l[k] for k in SPLAT_KEYS},
        }
    fake_rgb, fake_depth, Rt, K = gsn.generate_from_layout(gen_input, camera_params = camera_params)
    
    l = edited_layout
    SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
    gen_input = {
            'input': l['feature_grid'],
            'styles': {k: l[k] for k in SPLAT_KEYS},
        }
    fake_rgb2, fake_depth2, Rt2, K2 = gsn.generate_from_layout(gen_input, camera_params = camera_params)
    
    rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t h) w c').cpu().detach()
    rgb_current2 = rearrange(fake_rgb2, 'b t c h w -> (b t h) w c').cpu().detach()
    
    stacked_imgs.append(torch.hstack([rgb_current, rgb_current2, edited_labeled_blobs]))

n = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
animation_filename = 'walkthroughs/{}_moving_blobs.gif'.format(n)
imio.mimsave(animation_filename, stacked_imgs, fps=5)
display(HTML('<img src={}>'.format(animation_filename)))
print('Saved blob animation to {}\n'.format(animation_filename))


# # get trajectory
# trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
# actions = ['d','d','d','d','d','d','d','d','d','d','d','d', 'q']
# for action in actions:
#     camera_params = {'K': K_current}
    
#     with torch.no_grad():
#         fake_rgb, fake_depth, Rt, K = gsn(z, camera_params=camera_params)
    
#     trajectory['rgb'].append(fake_rgb)
#     trajectory['depth'].append(fake_depth)
#     trajectory['Rt'].append(Rt)
#     trajectory['K'].append(K)
    
#     rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t h) w c').cpu()
    
#     clear_output()
#     fig = plt.figure(figsize = (8, 8)) 
#     ax = fig.add_subplot(111)
#     ax.imshow(rgb_current, interpolation='bilinear')
#     ax.set_title('Current observation');
#     plt.axis('off')
#     plt.show()

#     if action == 'a':
#         # Turn left
#         Rt = rotate_n(n=-30.0).to(device)
#         Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
#     if action == 'd':
#         # Turn right
#         Rt = rotate_n(n=30.0).to(device)
#         Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
#     if action == 'w':
#         # Go forwardw
        
#         Rt_current = go_forward(Rt_current, step=opt.model_config.params.voxel_size / 0.6)
#     if action == 's':
#         # Go backward
#         Rt_current = go_backward(Rt_current, step=opt.model_config.params.voxel_size / 0.6)
#     if action == 'q':
#         break
        
# for key in trajectory.keys():
#     trajectory[key] = torch.cat(trajectory[key], dim=1)
    
# print('trajectory finished')
    
# # jitter camera pose a tiny amount to make sure each pose is unique
# # (to avoid problems with trajectory smoothing)
# trajectory['Rt'] = trajectory['Rt'] + torch.rand_like(trajectory['Rt']) * 1e-5

# # fit a smooth spline to the trajectory keypoints
# n_keypoints = len(trajectory['Rt'][0])
# new_Rts = get_smooth_trajectory(Rt=trajectory['Rt'][0], n_frames=5 * n_keypoints, subsample=3)
# n_steps = len(new_Rts)

# # render frames along new smooth trajectory
# resampled_trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
# for i in range(n_steps):
#     if i % 10 == 0:
#         print('Rendering frame {}/{}'.format(i, n_steps))
    
#     camera_params = {'K': K_current[:1, :1], 'Rt': new_Rts[i:i+1].to(device)}
    
#     with torch.no_grad():
#         fake_rgb, fake_depth, Rt, K = gsn(z, camera_params=camera_params)

#     resampled_trajectory['rgb'].append(fake_rgb)
#     resampled_trajectory['depth'].append(fake_depth)
#     resampled_trajectory['Rt'].append(Rt)
#     resampled_trajectory['K'].append(K)

# for key in resampled_trajectory.keys():
#     resampled_trajectory[key] = torch.cat(resampled_trajectory[key], dim=1)

# imgs = []
# for i in range(n_steps):
#     im = resampled_trajectory['rgb'][0, i].permute(1, 2, 0).detach().cpu()
#     imgs.append((im * 255).byte())

# # save gif with random unique name
# n = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
# animation_filename = 'walkthroughs/{}_camera_trajectory.gif'.format(n)
# imio.mimsave(animation_filename, imgs, fps=30)
# print('Saving animation to {}\n'.format(animation_filename))

# blob_filename = 'walkthroughs/{}_blobs.png'.format(n)
# im = Image.fromarray(blobs)
# im.save(blob_filename)
# print('Blob image saved to {}\n'.format(blob_filename))
