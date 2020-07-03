# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates masks of object models in the ground-truth poses."""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'lm',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # Tolerance used in the visibility test [mm].
  'delta': 15,  # 5 for ITODD, 15 for the other datasets.

  # Type of the renderer.
  'renderer_type': 'python',  # Options: 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = None
if p['dataset'] == 'tless':
  model_type = 'cad'
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:

  # Load scene GT.
  scene_gt_path = dp_split['scene_gt_tpath'].format(
    scene_id=scene_id)
  scene_gt = inout.load_scene_gt(scene_gt_path)

  # Load scene camera.
  scene_camera_path = dp_split['scene_camera_tpath'].format(
    scene_id=scene_id)
  scene_camera = inout.load_scene_camera(scene_camera_path)

  # Create folders for the output masks (if they do not exist yet).
  mask_dir_path = os.path.dirname(
    dp_split['mask_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_dir_path)

  mask_visib_dir_path = os.path.dirname(
    dp_split['mask_visib_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_visib_dir_path)

  # Initialize a renderer.
  misc.log('Initializing renderer...')
  width, height = dp_split['im_size']
  ren = renderer.create_renderer(
    width, height, renderer_type=p['renderer_type'], mode='depth')

  # Add object models.
  for obj_id in dp_model['obj_ids']:
    ren.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))

  im_ids = sorted(scene_gt.keys())
  for im_id in im_ids:

    if im_id % 100 == 0:
      misc.log(
        'Calculating masks - dataset: {} ({}, {}), scene: {}, im: {}'.format(
          p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id,
          im_id))

    K = scene_camera[im_id]['cam_K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Load depth image.
    depth_path = dp_split['depth_tpath'].format(
      scene_id=scene_id, im_id=im_id)
    depth_im = inout.load_depth(depth_path)
    depth_im *= scene_camera[im_id]['depth_scale']  # to [mm]
    dist_im = misc.depth_im_to_dist_im(depth_im, K)

    for gt_id, gt in enumerate(scene_gt[im_id]):

      # Render the depth image.
      depth_gt = ren.render_object(
        gt['obj_id'], gt['cam_R_m2c'], gt['cam_t_m2c'], fx, fy, cx, cy)['depth']

      # Convert depth image to distance image.
      dist_gt = misc.depth_im_to_dist_im(depth_gt, K)

      # Mask of the full object silhouette.
      mask = dist_gt > 0

      # Mask of the visible part of the object silhouette.
      mask_visib = visibility.estimate_visib_mask_gt(
        dist_im, dist_gt, p['delta'], visib_mode='bop19')

      # Save the calculated masks.
      mask_path = dp_split['mask_tpath'].format(
        scene_id=scene_id, im_id=im_id, gt_id=gt_id)
      inout.save_im(mask_path, 255 * mask.astype(np.uint8))

      mask_visib_path = dp_split['mask_visib_tpath'].format(
        scene_id=scene_id, im_id=im_id, gt_id=gt_id)
      inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))
