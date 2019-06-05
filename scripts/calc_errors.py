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
  'n_top': 1,  # Options: 0 = all, -1 = given by the number of GT poses.

  # Pose error function.
  # Options: 'vsd', 'adi', 'add', 'cou_mask_proj', 're', 'te'.
  'error_type': 're',

  # VSD parameters.
  'vsd_delta': 15,
  'vsd_tau': 20,
  'vsd_cost': 'step',  # Options: 'step', 'tlinear'.

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'cpp',  # Options: 'cpp', 'python'.

  # Paths to results for which the errors will be calculated.
  'result_paths': [
    r'C:\Users\tomho\th_data\cmp\projects\bop\test_data\results\hodan-iros15-dv1-nopso_ruapc',
    r'C:\Users\tomho\th_data\cmp\projects\bop\test_data\results\hodan-iros15-dv1-nopso_lm',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': r'C:\Users\tomho\th_data\msr\projects\pose6d\pose6d_scratch\datasets\bop',

  # A template of path to the output file with calculated errors.
  'out_errors_tpath': join(
    '{result_path}', '..', '..', 'eval', '{result_name}', '{error_sign}',
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

  # Parse the name of method and dataset from the folder name.
  result_name = os.path.basename(result_path)
  result_info = result_name.split('_')
  method = result_info[0]
  dataset = result_info[1]
  test_type = result_info[2] if len(result_info) > 2 else None

  # Select the type of camera and object models.
  if dataset == 'tless':
    cam_type = test_type
    if p['error_type'] in ['adi', 'add']:
      model_type = 'cad_subdivided'
    else:
      model_type = 'cad'
  else:
    cam_type = None
    model_type = None

  # Load dataset parameters.
  dp = dataset_params.get_dataset_params(
    p['datasets_path'], dataset, model_type=model_type, test_type=test_type,
    cam_type=cam_type)

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
    ren = renderer.create_renderer(dp['test_im_size'], p['renderer_type'])

    for obj_id in dp['obj_ids']:
      ren.add_object(obj_id, dp['model_tpath'].format(obj_id))

  # Directories with results for individual scenes.
  scene_dirs = sorted(
    [d for d in glob.glob(os.path.join(result_path, '*')) if os.path.isdir(d)])

  for scene_dir in scene_dirs:
    scene_id = int(os.path.basename(scene_dir))

    # Load info and GT poses for the current scene.
    scene_info = inout.load_info(
      dp['test_info_tpath'].format(scene_id=scene_id))
    scene_gt = inout.load_gt(dp['test_gt_tpath'].format(scene_id=scene_id))

    res_paths = sorted(glob.glob(os.path.join(scene_dir, '*.yml')))
    errs = []
    im_id = -1
    depth_im = None
    res_count = len(res_paths)
    for res_id, res_path in enumerate(res_paths):
      t_start = time.time()

      # Parse image ID and object ID from the filename.
      filename = os.path.basename(res_path).split('.')[0]
      im_id_prev = im_id
      im_id, obj_id = map(int, filename.split('_'))

      # Log state.
      if res_id % 100 == 0:
        dataset_str = dataset
        if test_type is not None:
          dataset_str += ' - {}'.format(test_type)
        misc.log(
          'Calculating error - error: {}, method: {}, dataset: {}, scene: {}, '
          '{}/{}.'.format(p['error_type'], method, dataset_str, scene_id,
                          res_id, res_count))

      # Load depth image if VSD is selected as the pose error function.
      if p['error_type'] == 'vsd' and im_id != im_id_prev:
        depth_path = dp['test_depth_tpath'].format(scene_id, im_id)
        depth_im = inout.load_depth(depth_path)
        depth_im *= dp['cam']['depth_scale']  # Convert to [mm].

      # Load camera matrix.
      K = None
      if p['error_type'] in ['vsd', 'cou_mask_proj']:
        K = scene_info[im_id]['cam_K']

      # Load pose estimates.
      res = inout.load_bop_results(res_path)
      ests = res['ests']

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
      # object.
      for est_id, est in ests_sorted:
        R_e = est['R']
        t_e = est['t']

        errs_gts = {}  # Errors w.r.t. GT poses of the same object.
        for gt_id, gt in enumerate(scene_gt[im_id]):
          if gt['obj_id'] != obj_id:
            continue

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

          errs_gts[gt_id] = e

        # Save the calculated errors.
        errs.append({
          'im_id': im_id,
          'obj_id': obj_id,
          'est_id': est_id,
          'score': est['score'],
          'errors': errs_gts
        })

    # Save the errors to a YAML file.
    errors_path = p['out_errors_tpath'].format(
      result_path=result_path, result_name=result_name,
      error_sign=error_sign, scene_id=scene_id)
    misc.ensure_dir(os.path.dirname(errors_path))
    misc.log('Saving errors to: {}'.format(errors_path))
    inout.save_errors(errors_path, errs)

misc.log('Done.')
