# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in the ground-truth poses."""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'lm',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # File with a list of estimation targets used to determine the set of images
  # for which the GT poses will be visualized. The file is assumed to be stored
  # in the dataset folder. None = all images.
  # 'targets_filename': 'test_targets_bop19.json',
  'targets_filename': None,

  # Select ID's of scenes, images and GT poses to be processed.
  # Empty list [] means that all ID's will be used.
  'scene_ids': [],
  'im_ids': [],
  'gt_ids': [],
  
  # Indicates whether to render RGB images.
  'vis_rgb': True,
  
  # Indicates whether to resolve visibility in the rendered RGB images (using
  # depth renderings). If True, only the part of object surface, which is not
  # occluded by any other modeled object, is visible. If False, RGB renderings
  # of individual objects are blended together.
  'vis_rgb_resolve_visib': True,
  
  # Indicates whether to save images of depth differences.
  'vis_depth_diff': False,
  
  # Whether to use the original model color.
  'vis_orig_color': False,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'python',  # Options: 'cpp', 'python'.
  
  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
  
  # Folder for output visualisations.
  'vis_path': os.path.join(config.output_path, 'vis_gt_poses'),
  
  # Path templates for output images.
  'vis_rgb_tpath': os.path.join(
    '{vis_path}', '{dataset}', '{split}', '{scene_id:06d}', '{im_id:06d}.jpg'),
  'vis_depth_diff_tpath': os.path.join(
    '{vis_path}', '{dataset}', '{split}', '{scene_id:06d}',
    '{im_id:06d}_depth_diff.jpg'),
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = 'eval'  # None = default.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Load colors.
colors_path = os.path.join(
  os.path.dirname(visualization.__file__), 'colors.json')
colors = inout.load_json(colors_path)

# Subset of images for which the ground-truth poses will be rendered.
if p['targets_filename'] is not None:
  targets = inout.load_json(
    os.path.join(dp_split['base_path'], p['targets_filename']))
  scene_im_ids = {}
  for target in targets:
    scene_im_ids.setdefault(
      target['scene_id'], set()).add(target['im_id'])
else:
  scene_im_ids = None

# List of considered scenes.
scene_ids_curr = dp_split['scene_ids']
if p['scene_ids']:
  scene_ids_curr = set(scene_ids_curr).intersection(p['scene_ids'])

# Rendering mode.
renderer_modalities = []
if p['vis_rgb']:
  renderer_modalities.append('rgb')
if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
  renderer_modalities.append('depth')
renderer_mode = '+'.join(renderer_modalities)

# Create a renderer.
width, height = dp_split['im_size']
ren = renderer.create_renderer(
  width, height, p['renderer_type'], mode=renderer_mode, shading='flat')

# Load object models.
models = {}
for obj_id in dp_model['obj_ids']:
  misc.log('Loading 3D model of object {}...'.format(obj_id))
  model_path = dp_model['model_tpath'].format(obj_id=obj_id)
  model_color = None
  if not p['vis_orig_color']:
    model_color = tuple(colors[(obj_id - 1) % len(colors)])
  ren.add_object(obj_id, model_path, surf_color=model_color)

for scene_id in scene_ids_curr:

  # Load scene info and ground-truth poses.
  scene_camera = inout.load_scene_camera(
    dp_split['scene_camera_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_scene_gt(
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))

  # List of considered images.
  if scene_im_ids is not None:
    im_ids = scene_im_ids[scene_id]
  else:
    im_ids = sorted(scene_gt.keys())
  if p['im_ids']:
    im_ids = set(im_ids).intersection(p['im_ids'])

  # Render the object models in the ground-truth poses in the selected images.
  for im_counter, im_id in enumerate(im_ids):
    if im_counter % 10 == 0:
      misc.log(
        'Visualizing GT poses - dataset: {}, scene: {}, im: {}/{}'.format(
          p['dataset'], scene_id, im_counter, len(im_ids)))

    K = scene_camera[im_id]['cam_K']

    # List of considered ground-truth poses.
    gt_ids_curr = range(len(scene_gt[im_id]))
    if p['gt_ids']:
      gt_ids_curr = set(gt_ids_curr).intersection(p['gt_ids'])

    # Collect the ground-truth poses.
    gt_poses = []
    for gt_id in gt_ids_curr:
      gt = scene_gt[im_id][gt_id]
      gt_poses.append({
        'obj_id': gt['obj_id'],
        'R': gt['cam_R_m2c'],
        't': gt['cam_t_m2c'],
        'text_info': [
          {'name': '', 'val': '{}:{}'.format(gt['obj_id'], gt_id), 'fmt': ''}
        ]
      })

    # Load the color and depth images and prepare images for rendering.
    rgb = None
    if p['vis_rgb']:
      if 'rgb' in dp_split['im_modalities']:
        rgb = inout.load_im(dp_split['rgb_tpath'].format(
          scene_id=scene_id, im_id=im_id))[:, :, :3]
      elif 'gray' in dp_split['im_modalities']:
        gray = inout.load_im(dp_split['gray_tpath'].format(
          scene_id=scene_id, im_id=im_id))
        rgb = np.dstack([gray, gray, gray])
      else:
        raise ValueError('RGB nor gray images are available.')

    depth = None
    if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
      depth = inout.load_depth(dp_split['depth_tpath'].format(
        scene_id=scene_id, im_id=im_id))
      depth *= scene_camera[im_id]['depth_scale']  # Convert to [mm].

    # Path to the output RGB visualization.
    vis_rgb_path = None
    if p['vis_rgb']:
      vis_rgb_path = p['vis_rgb_tpath'].format(
        vis_path=p['vis_path'], dataset=p['dataset'], split=p['dataset_split'],
        scene_id=scene_id, im_id=im_id)

    # Path to the output depth difference visualization.
    vis_depth_diff_path = None
    if p['vis_depth_diff']:
      vis_depth_diff_path = p['vis_depth_diff_tpath'].format(
        vis_path=p['vis_path'], dataset=p['dataset'], split=p['dataset_split'],
        scene_id=scene_id, im_id=im_id)

    # Visualization.
    visualization.vis_object_poses(
      poses=gt_poses, K=K, renderer=ren, rgb=rgb, depth=depth,
      vis_rgb_path=vis_rgb_path, vis_depth_diff_path=vis_depth_diff_path,
      vis_rgb_resolve_visib=p['vis_rgb_resolve_visib'])

misc.log('Done.')
