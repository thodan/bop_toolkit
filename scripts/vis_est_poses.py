# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in pose estimates saved in the BOP format."""

import os
import glob
from os.path import join

from bop_toolkit import dataset_params, inout, misc, visualization


# PARAMETERS.
################################################################################
p = {
  # Top N pose estimates (with the highest score) to be displayed for each
  # object in each image.
  'n_top': -1,  # 0 = all estimates, -1 = given by the number of GT poses.

  # Indicates whether to render RGB image.
  'vis_rgb': True,

  # Indicates whether to resolve visibility in the rendered RGB images (using
  # depth renderings). If True, only the part of object surface, which is not
  # occluded by any other modeled object, is visible. If False, RGB renderings
  # of individual objects are blended together.
  'vis_rgb_resolve_visib': True,

  # Indicates whether to render depth image.
  'vis_depth_diff': False,

  # If to use the original model color.
  'vis_orig_color': False,

  # Paths to results in the BOP format (see docs/bop_results_format.md).
  'result_paths': [
    r'/PATH/TO/RESULTS/IN/BOP/FORMAT',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': r'/PATH/TO/BOP/DATASETS',
  
  # Folder for output visualisations.
  'vis_folder_path': r'/PATH/TO/OUTPUT/FOLDER',
  
  # Path templates for output images.
  'vis_rgb_tpath': join(
    '{vis_folder_path}', 'vis_est_poses', '{result_name}', '{scene_id:06d}',
    '{im_id:06d}_{obj_id:06d}.jpg'),
  'vis_depth_diff_tpath': join(
    '{vis_folder_path}', 'vis_est_poses', '{result_name}', '{scene_id:06d}',
    '{im_id:06d}_{obj_id:06d}_depth_diff.jpg'),
}
################################################################################


# Load colors.
colors_path = join(os.path.dirname(visualization.__file__), 'colors.yml')
colors = inout.load_yaml(colors_path)

for result_path in p['result_paths']:
  misc.log('Processing: ' + result_path)

  # Parse info about the method and the dataset from the folder name.
  result_name = os.path.basename(result_path)
  result_info = result_name.split('_')
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None

  # Object models type.
  model_type = None  # None = default.
  if dataset == 'tless':
    model_type = 'cad_subdivided'

  # Load dataset parameters.
  dp = dataset_params.get_dataset_params(
    p['datasets_path'], dataset, model_type=model_type, train_type=split_type,
    val_type=split_type, test_type=split_type, cam_type=split_type)

  # Load object models.
  models = {}
  for obj_id in dp['obj_ids']:
    models[obj_id] = inout.load_ply(dp['model_tpath'].format(obj_id=obj_id))

  # Colors used for the text labels and (optionally) for the object surface.
  model_colors = {}
  for obj_id in dp['obj_ids']:
    if p['vis_orig_color']:
      model_colors[obj_id] = (1.0, 1.0, 1.0)
    else:
      c_id = (obj_id - 1) % len(colors)
      model_colors[obj_id] = tuple(colors[c_id])

  # Directories with results for individual scenes.
  scene_dirs = sorted([d for d in glob.glob(os.path.join(result_path, '*'))
                       if os.path.isdir(d)])

  for scene_dir in scene_dirs:
    scene_id = int(os.path.basename(scene_dir))

    # Load info and ground-truth poses for the current scene.
    scene_info = inout.load_info(
      dp[split + '_info_tpath'].format(scene_id=scene_id))
    scene_gt = inout.load_gt(
      dp[split + '_gt_tpath'].format(scene_id=scene_id))

    # Paths to the YAML files with results.
    res_paths = sorted(
      glob.glob(os.path.join(scene_dir, '*.yml')) +
      glob.glob(os.path.join(scene_dir, '*.yaml'))
    )

    # Visualize the results one by one.
    im_id = -1
    depth_im = None
    for res_id, res_path in enumerate(res_paths):

      # Parse image ID and object ID from the filename.
      filename = os.path.basename(res_path).split('.')[0]
      im_id_prev = im_id
      im_id, obj_id = map(int, filename.split('_'))

      if res_id % 10 == 0:
        split_type_str = ' - ' + split_type if split_type is not None else ''
        misc.log(
          'Visualizing pose estimates - method: {}, dataset: {}{}, scene: {}, '
          'im: {}'.format(method, dataset, split_type_str, scene_id, im_id))

      # Load camera matrix.
      K = scene_info[im_id]['cam_K']

      # Load pose estimates.
      ests = inout.load_bop_results(res_path)['ests']

      # Sort the estimates by score (in descending order).
      ests_sorted = sorted(enumerate(ests), key=lambda x: x[1]['score'],
                           reverse=True)

      # Select the number of top estimated poses to visualize.
      if p['n_top'] == 0:  # All estimates are considered.
        n_top_curr = None
      elif p['n_top'] == -1:  # Given by the number of GT poses.
        n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
        n_top_curr = n_gt
      else:  # Specified by the parameter n_top.
        n_top_curr = p['n_top']
      ests_sorted = ests_sorted[slice(0, n_top_curr)]

      # Get list of poses to visualize.
      poses = [e[1] for e in ests_sorted]
      for pose in poses:
        pose['obj_id'] = obj_id

        # Text info to write on the image at the pose estimate.
        pose['text_info'] = [
          {'name': '', 'val': pose['score'], 'fmt': ':.2f'}
        ]

      # Load the color and depth images and prepare images for rendering.
      rgb = None
      if p['vis_rgb']:
        rgb = inout.load_im(dp[split + '_rgb_tpath'].format(
          scene_id=scene_id, im_id=im_id))

      depth = None
      if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
        depth = inout.load_depth(dp[split + '_depth_tpath'].format(
          scene_id=scene_id, im_id=im_id))
        depth *= dp['cam']['depth_scale']  # Convert to [mm].

      # Path to the output RGB visualization.
      vis_rgb_path = None
      if p['vis_rgb']:
        vis_rgb_path = p['vis_rgb_tpath'].format(
          vis_folder_path=p['vis_folder_path'], result_name=result_name,
          scene_id=scene_id, im_id=im_id, obj_id=obj_id)

      # Path to the output depth difference visualization.
      vis_depth_diff_path = None
      if p['vis_depth_diff']:
        vis_depth_diff_path = p['vis_depth_diff_tpath'].format(
          vis_folder_path=p['vis_folder_path'], result_name=result_name,
          scene_id=scene_id, im_id=im_id, obj_id=obj_id)

      # Visualization.
      visualization.vis_object_poses(
        poses=poses, K=K, models=models, model_colors=model_colors,
        rgb=rgb, depth=depth, vis_rgb_path=vis_rgb_path,
        vis_depth_diff_path=vis_depth_diff_path,
        vis_orig_color=p['vis_orig_color'],
        vis_rgb_resolve_visib=p['vis_rgb_resolve_visib'])

misc.log('Done.')
