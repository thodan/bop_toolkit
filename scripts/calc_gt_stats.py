# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates statistics of individual ground-truth poses.

See docs/bop_datasets_format.md for documentation of the calculated statistics.

The statistics are saved in folder "{train,val,test}_gt_stats" in the main
folder of the selected dataset.
"""

import os
from os.path import join
import numpy as np

from bop_toolkit import dataset_params, inout, misc, renderer_py, visibility,\
  visualization


# PARAMETERS.
################################################################################
p = {
  # Options: 'lm', 'lmo', 'tless', 'tudl', 'ruapc', 'icmi', 'icbin'.
  'dataset': 'tless',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Name of file with a list of image ID's to be used. The file is assumed to be
  # stored in the dataset folder. None = all images are used for the evaluation.
  'im_subset_filename': 'test_set_v1.yml',

  # Whether to save visualizations of visibility masks.
  'vis_visibility_masks': True,

  # Tolerance used in the visibility test [mm].
  'delta': 15,

  # Folder containing the BOP datasets.
  'datasets_path': r'/path/to/bop/datasets',

  # Folder for output visualisations.
  'vis_folder_path': r'/path/to/output/folder',

  # Path templates for output images.
  'vis_tpath': join(
    '{vis_folder_path}', 'vis_gt_visib_delta={delta}', '{dataset}', '{split}',
    '{scene_id:06d}', '{im_id:06d}_{gt_id:06d}.jpg'),
}
################################################################################


# Type of object models and images.
model_type = None  # None = default.
data_type = None  # None = default.
if p['dataset'] == 'tless':
  model_type = 'cad_subdivided'
  data_type = 'primesense'

# Load dataset parameters.
dp = dataset_params.get_dataset_params(
  p['datasets_path'], p['dataset'], model_type=model_type, train_type=data_type,
  val_type=data_type, test_type=data_type, cam_type=data_type)

# Subset of images to be considered.
if p['im_subset_filename'] is not None:
  im_ids_sets = inout.load_yaml(
    os.path.join(dp['base_path'], p['im_subset_filename']))
else:
  im_ids_sets = None

# Load 3D object models.
misc.log('Loading object models...')
models = {}
for obj_id in dp['obj_ids']:
  models[obj_id] = inout.load_ply(dp['model_tpath'].format(obj_id=obj_id))

for scene_id in dp['scene_ids']:

  # Load scene info and ground-truth poses.
  scene_info = inout.load_info(
    dp[p['dataset_split'] + '_info_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_gt(
    dp[p['dataset_split'] + '_gt_tpath'].format(scene_id=scene_id))

  # Considered subset of images for the current scene.
  if im_ids_sets is not None:
    im_ids = im_ids_sets[scene_id]
  else:
    im_ids = sorted(scene_gt.keys())

  scene_gt_stats = {}
  for im_id in im_ids:
    misc.log('Calculating GT stats - dataset: {}, scene: {}, im: {}/{}'.format(
      p['dataset'], scene_id, im_id, len(im_ids)))

    # Load depth image.
    depth = inout.load_depth(dp[p['dataset_split'] + '_depth_tpath'].format(
      scene_id=scene_id, im_id=im_id))
    depth *= dp['cam']['depth_scale']  # Convert to [mm].

    K = scene_info[im_id]['cam_K']
    im_size = (depth.shape[1], depth.shape[0])

    scene_gt_stats[im_id] = []
    for gt_id, gt in enumerate(scene_gt[im_id]):

      # Render depth image of the object model in the ground-truth pose.
      depth_gt = renderer_py.render(
        models[gt['obj_id']], im_size, K, gt['cam_R_m2c'], gt['cam_t_m2c'],
        mode='depth')

      # Convert depth images to distance images.
      dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
      dist_im = misc.depth_im_to_dist_im(depth, K)

      # Estimation of the visibility mask.
      visib_gt = visibility.estimate_visib_mask_gt(dist_im, dist_gt, p['delta'])

      # Visible surface fraction.
      obj_mask_gt = dist_gt > 0
      px_count_valid = np.sum(dist_im[obj_mask_gt] > 0)
      px_count_visib = visib_gt.sum()
      px_count_all = obj_mask_gt.sum()
      if px_count_all > 0:
        visib_fract = px_count_visib / float(px_count_all)
      else:
        visib_fract = 0.0

      # Bounding box of the object projection
      ys, xs = obj_mask_gt.nonzero()
      bbox = misc.calc_2d_bbox(xs, ys, im_size)

      # Bounding box of the visible surface part.
      bbox_visib = [-1, -1, -1, -1]
      if px_count_visib > 0:
        ys, xs = visib_gt.nonzero()
        bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)

      # Store the calculated stats.
      scene_gt_stats[im_id].append({
        'px_count_all': int(px_count_all),
        'px_count_visib': int(px_count_visib),
        'px_count_valid': int(px_count_valid),
        'visib_fract': float(visib_fract),
        'bbox_obj': [int(e) for e in bbox],
        'bbox_visib': [int(e) for e in bbox_visib]
      })

      # Visualization of the visibility mask.
      if p['vis_visibility_masks']:

        depth_im_vis = visualization.depth_for_vis(depth, 0.2, 1.0)
        depth_im_vis = np.dstack([depth_im_vis] * 3)

        visib_gt_vis = visib_gt.astype(np.float)
        zero_ch = np.zeros(visib_gt_vis.shape)
        visib_gt_vis = np.dstack([zero_ch, visib_gt_vis, zero_ch])

        vis = 0.5 * depth_im_vis + 0.5 * visib_gt_vis
        vis[vis > 1] = 1

        vis_path = p['vis_tpath'].format(
          vis_folder_path=p['vis_folder_path'], delta=p['delta'],
          dataset=p['dataset'], split=p['dataset_split'], scene_id=scene_id,
          im_id=im_id, gt_id=gt_id)
        misc.ensure_dir(os.path.dirname(vis_path))
        inout.save_im(vis_path, vis)

    break

  # Save the stats for the current scene.
  scene_gt_stats_path = dp[p['dataset_split'] + '_gt_stats_tpath'].format(
    scene_id=scene_id, delta=p['delta'])
  misc.ensure_dir(os.path.dirname(scene_gt_stats_path))
  inout.save_yaml(scene_gt_stats_path, scene_gt_stats)
