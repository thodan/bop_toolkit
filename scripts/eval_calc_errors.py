# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the error of 6D object pose estimates."""

import os
from os.path import join
import glob
import time
import argparse

from bop_toolkit import inout, pose_error, misc, renderer, dataset_params


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
  # Top N pose estimates (with the highest score) to be evaluated for each
  # object class in each image.
  # Options: 0 = all, -1 = given by the number of GT poses.
  'n_top': 1,

  # Pose error function.
  # Options: 'vsd', 'adi', 'add', 'cou_mask_proj', 'rete', 're', 'te'.
  'error_type': 'vsd',

  # VSD parameters.
  'vsd_delta': 15,
  'vsd_tau': 20,
  'vsd_cost': 'step',  # Options: 'step', 'tlinear'.

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'cpp',  # Options: 'cpp', 'python'.

  # Paths to results for which the errors will be calculated.
  'result_paths': [
    r'/path/to/results/in/bop/format',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': r'/path/to/bop/datasets',

  # Folder in which the calculated errors will be saved.
  'out_errors_dir': r'/path/to/output/folder',

  # Template of path to the output file with calculated errors.
  'out_errors_tpath': join('{out_errors_dir}', '{result_name}', '{error_sign}',
                           'errors_{scene_id:06d}.yml')
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n_top', default=p['n_top'])
parser.add_argument('--error_type', default=p['error_type'])
parser.add_argument('--vsd_delta', default=p['vsd_delta'])
parser.add_argument('--vsd_tau', default=p['vsd_tau'])
parser.add_argument('--vsd_cost', default=p['vsd_cost'])
parser.add_argument('--renderer_type', default=p['renderer_type'])
parser.add_argument('--result_paths', default=','.join(p['result_paths']),
                    help='Comma-separated paths to results.')
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--out_errors_dir', default=p['out_errors_dir'])
parser.add_argument('--out_errors_tpath', default=p['out_errors_tpath'])
args = parser.parse_args()

p['result_paths'] = args.result_paths.split(',')
p['n_top'] = int(args.n_top)
p['error_type'] = str(args.error_type)
p['vsd_delta'] = float(args.vsd_delta)
p['vsd_tau'] = float(args.vsd_tau)
p['vsd_cost'] = str(args.vsd_cost)
p['renderer_type'] = str(args.renderer_type)
p['datasets_path'] = str(args.datasets_path)
p['out_errors_dir'] = str(args.out_errors_dir)
p['out_errors_tpath'] = str(args.out_errors_tpath)

misc.log('----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('----------')

# Error calculation.
# ------------------------------------------------------------------------------
# Error signature.
error_sign = 'error=' + p['error_type'] + '_ntop=' + str(p['n_top'])
if p['error_type'] == 'vsd':
  error_sign += '_delta={}_tau={}_cost={}'.format(
    int(p['vsd_delta']), int(p['vsd_tau']), p['vsd_cost'])

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

  # Select the type of camera and object models.
  if dataset == 'tless':
    cam_type = split_type
    if p['error_type'] in ['adi', 'add']:
      model_type = 'cad_subdivided'
    else:
      model_type = 'cad'
  else:
    cam_type = None
    model_type = None

  # Load dataset parameters.
  dp = dataset_params.get_dataset_params(
    p['datasets_path'], dataset, model_type=model_type, train_type=split_type,
    val_type=split_type, test_type=split_type, cam_type=split_type)

  # Load object models.
  models = {}
  if p['error_type'] in ['add', 'adi']:
    misc.log('Loading object models...')
    for obj_id in dp['obj_ids']:
      models[obj_id] = inout.load_ply(dp['model_tpath'].format(obj_id))

  # Initialize renderer.
  ren = None
  if p['error_type'] in ['vsd', 'cou_mask_proj']:
    misc.log('Initializing renderer...')
    ren = renderer.create_renderer(dp[split + '_im_size'], p['renderer_type'])
    for obj_id in dp['obj_ids']:
      ren.add_object(obj_id, dp['model_tpath'].format(obj_id))

  # Folders with pose estimates for individual scenes.
  scene_dirs = sorted(
    [d for d in glob.glob(os.path.join(result_path, '*')) if os.path.isdir(d)])

  for scene_dir in scene_dirs:
    scene_id = int(os.path.basename(scene_dir))

    # Load info and GT poses for the current scene.
    scene_info = inout.load_info(
      dp[split + '_info_tpath'].format(scene_id=scene_id))
    scene_gt = inout.load_gt(dp[split + '_gt_tpath'].format(scene_id=scene_id))

    # Paths to files with pose estimates -- there is one file per [image ID,
    # object ID] pair.
    res_paths = sorted(glob.glob(os.path.join(scene_dir, '*.yml')))

    scene_errs = []
    im_id = -1
    depth_im = None
    for res_id, res_path in enumerate(res_paths):
      t_start = time.time()

      # Parse image ID and object ID from the filename.
      filename = os.path.basename(res_path).split('.')[0]
      im_id_prev = im_id
      im_id, obj_id = map(int, filename.split('_'))

      # Log state.
      if res_id % 100 == 0:
        dataset_str = dataset
        if split_type is not None:
          dataset_str += ' - {}'.format(split_type)
        misc.log(
          'Calculating error - error: {}, method: {}, dataset: {}, scene: {}, '
          '{}/{}.'.format(p['error_type'], method, dataset_str, scene_id,
                          res_id, len(res_paths)))

      # Load the depth image if VSD is selected as the pose error function and
      # it has not been loaded for the current frame yet.
      if p['error_type'] == 'vsd' and im_id != im_id_prev:
        depth_path = dp[split + '_depth_tpath'].format(scene_id, im_id)
        depth_im = inout.load_depth(depth_path)
        depth_im *= dp['cam']['depth_scale']  # Convert to [mm].

      # Load camera matrix if needed.
      K = None
      if p['error_type'] in ['vsd', 'cou_mask_proj']:
        K = scene_info[im_id]['cam_K']

      # Load pose estimates.
      ests = inout.load_bop_results(res_path)['ests']

      # Sort the estimates by score (in descending order).
      ests_sorted = sorted(
        enumerate(ests), key=lambda x: x[1]['score'], reverse=True)

      # Select the required number of top estimated poses.
      if p['n_top'] == 0:  # All estimates are considered.
        n_top_curr = None
      elif p['n_top'] == -1:  # Given by the number of GT poses.
        n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
        n_top_curr = n_gt
      else:
        n_top_curr = p['n_top']
      ests_sorted = ests_sorted[slice(0, n_top_curr)]

      # Calculate error of each pose estimate w.r.t. all GT poses of the same
      # object class.
      for est_id, est in ests_sorted:

        # Estimated pose.
        R_e = est['R']
        t_e = est['t']

        errs = {}  # Errors w.r.t. GT poses of the same object class.
        for gt_id, gt in enumerate(scene_gt[im_id]):
          if gt['obj_id'] != obj_id:
            continue

          # Ground-truth pose.
          R_g = gt['cam_R_m2c']
          t_g = gt['cam_t_m2c']

          if p['error_type'] == 'vsd':
            e = [pose_error.vsd(R_e, t_e, R_g, t_g, depth_im, K, p['vsd_delta'],
                                p['vsd_tau'], ren, obj_id, p['vsd_cost'])]
          elif p['error_type'] == 'add':
            e = [pose_error.add(R_e, t_e, R_g, t_g, models[obj_id]['pts'])]
          elif p['error_type'] == 'adi':
            e = [pose_error.adi(R_e, t_e, R_g, t_g, models[obj_id]['pts'])]
          elif p['error_type'] == 'cou_mask_proj':
            e = [pose_error.cou_mask_proj(R_e, t_e, R_g, t_g, K, ren, obj_id)]
          elif p['error_type'] == 'rete':
            e = [pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)]
          elif p['error_type'] == 're':
            e = [pose_error.re(R_e, R_g)]
          elif p['error_type'] == 'te':
            e = [pose_error.te(t_e, t_g)]
          else:
            raise ValueError('Unknown pose error function.')

          errs[gt_id] = e

        # Save the calculated errors.
        scene_errs.append({
          'im_id': im_id,
          'obj_id': obj_id,
          'est_id': est_id,
          'score': est['score'],
          'errors': errs
        })

    # Save the calculated errors to a YAML file.
    errors_path = p['out_errors_tpath'].format(
      out_errors_dir=p['out_errors_dir'], result_name=result_name,
      error_sign=error_sign, scene_id=scene_id)
    misc.ensure_dir(os.path.dirname(errors_path))
    misc.log('Saving errors to: {}'.format(errors_path))
    inout.save_errors(errors_path, scene_errs)

misc.log('Done.')
