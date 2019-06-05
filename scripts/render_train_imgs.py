# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Renders RGB-D images of an object model."""

import os
from os.path import join
import numpy as np
import cv2

from bop_toolkit import dataset_params, inout, misc, renderer_py, view_sampler


# PARAMETERS.
################################################################################
# Options: 'lm', 'lmo', 'tless', 'tudl', 'ruapc', 'icmi', 'icbin'
dataset = 'lm'

# Radii of view spheres from which to render the objects.
if dataset == 'lm':
  radii = [400]  # There are only 3 occurrences under 400 mm.
elif dataset == 'tless':
  radii = [650]
elif dataset == 'tudl':
  radii = [850]
elif dataset == 'ruapc':
  radii = [590]
elif dataset == 'icmi':
  radii = [500]
elif dataset == 'icbin':
  radii = [450]
else:
  raise ValueError('Unknown dataset.')

# Type of object models and camera.
model_type = None
cam_type = None
if dataset == 'tless':
  model_type = 'reconst'
  cam_type = 'primesense'

# Objects to render ([] = all objects from the specified dataset).
obj_ids = []

# Minimum required number of views on the whole view sphere. The final number of
# views depends on the sampling method.
min_n_views = 1000

# Rendering parameters.
clip_near = 10  # [mm]
clip_far = 10000  # [mm]
ambient_weight = 0.3  # Weight of ambient light [0, 1]
shading = 'phong'  # 'flat', 'phong'

# Super-sampling anti-aliasing (SSAA).
# Ref: https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
# The RGB image is rendered at ssaa_fact times higher resolution and then
# down-sampled to the required resolution.
ssaa_fact = 4

# Folder containing the BOP datasets.
datasets_path = r'/PATH/TO/BOP/DATASETS'

# Folder for the rendered images.
out_folder = r'/PATH/TO/OUTPUT/FOLDER'

# Output path templates.
out_rgb_tpath =\
  join('{out_folder}', '{obj_id:06d}', 'rgb', '{im_id:06d}.png')
out_depth_tpath =\
  join('{out_folder}', '{obj_id:06d}', 'depth', '{im_id:06d}.png')
out_scene_info_tpath =\
  join('{out_folder}', '{obj_id:06d}', 'info.yml')
out_scene_gt_tpath =\
  join('{out_folder}', '{obj_id:06d}', 'gt.yml')
out_views_vis_tpath =\
  join('{out_folder}', '{obj_id:06d}', 'views_radius={radius}.ply')
################################################################################


# Load dataset parameters.
dp = dataset_params.get_dataset_params(
  datasets_path, dataset, model_type=model_type, cam_type=cam_type)

if not obj_ids:
  obj_ids = dp['obj_ids']

# Image size and K for the RGB image (potentially with SSAA).
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['cam']['im_size']]
K_rgb = dp['cam']['K'] * ssaa_fact

for obj_id in obj_ids:

  # Prepare output folders.
  misc.ensure_dir(os.path.dirname(out_rgb_tpath.format(
    out_folder=out_folder, obj_id=obj_id, im_id=0)))
  misc.ensure_dir(os.path.dirname(out_depth_tpath.format(
    out_folder=out_folder, obj_id=obj_id, im_id=0)))

  # Load model.
  model_path = dp['model_tpath'].format(obj_id=obj_id)
  model = inout.load_ply(model_path)

  # Load model texture.
  if 'texture_file' in model:
    model_texture_path =\
      join(os.path.dirname(model_path), model['texture_file'])
    model_texture = inout.load_im(model_texture_path)
  else:
    model_texture = None

  obj_info = {}
  obj_gt = {}
  im_id = 0
  for radius in radii:
    # Sample views.
    view_sampler_mode = 'hinterstoisser'  # 'hinterstoisser' or 'fibonacci'.
    views, views_level = view_sampler.sample_views(
      min_n_views, radius, dp['test_azimuth_range'], dp['test_elev_range'],
      view_sampler_mode)

    misc.log('Sampled views: ' + str(len(views)))
    out_views_vis_path = out_views_vis_tpath.format(
      out_folder=out_folder, obj_id=obj_id, radius=radius)
    view_sampler.save_vis(out_views_vis_path, views, views_level)

    # Render the object model from all views.
    for view_id, view in enumerate(views):
      if view_id % 10 == 0:
        misc.log('Rendering - obj: {}, radius: {}, view: {}/{}'.format(
          obj_id, radius, view_id, len(views)))

      # Render depth image.
      depth = renderer_py.render(
        model, dp['cam']['im_size'], dp['cam']['K'], view['R'], view['t'],
        clip_near, clip_far, mode='depth')

      # Convert depth so it is in the same units as the real test images.
      depth /= dp['cam']['depth_scale']

      # Render RGB image.
      rgb = renderer_py.render(
        model, im_size_rgb, K_rgb, view['R'], view['t'], clip_near, clip_far,
        texture=model_texture, ambient_weight=ambient_weight, shading=shading,
        mode='rgb')

      # The OpenCV function was used for rendering of the training images
      # provided for the SIXD Challenge 2017.
      rgb = cv2.resize(rgb, dp['cam']['im_size'], interpolation=cv2.INTER_AREA)
      # rgb = scipy.misc.imresize(rgb, par['cam']['im_size'][::-1], 'bicubic')

      # Save the rendered images.
      out_rgb_path = out_rgb_tpath.format(
        out_folder=out_folder, obj_id=obj_id, im_id=im_id)
      inout.save_im(out_rgb_path, rgb)
      out_depth_path = out_depth_tpath.format(
        out_folder=out_folder, obj_id=obj_id, im_id=im_id)
      inout.save_depth(out_depth_path, depth)

      # Get 2D bounding box of the object model at the ground truth pose.
      ys, xs = np.nonzero(depth > 0)
      obj_bb = misc.calc_2d_bbox(xs, ys, dp['cam']['im_size'])

      obj_info[im_id] = {
        'cam_K': dp['cam']['K'].flatten().tolist(),
        'depth_scale': dp['cam']['depth_scale'],
        'view_level': int(views_level[view_id])
      }

      obj_gt[im_id] = [{
        'cam_R_m2c': view['R'].flatten().tolist(),
        'cam_t_m2c': view['t'].flatten().tolist(),
        'obj_bb': [int(x) for x in obj_bb],
        'obj_id': int(obj_id)
      }]

      im_id += 1

  # Save metadata.
  inout.save_yaml(out_scene_info_tpath.format(
    out_folder=out_folder, obj_id=obj_id), obj_info)
  inout.save_yaml(out_scene_gt_tpath.format(
    out_folder=out_folder, obj_id=obj_id), obj_gt)
