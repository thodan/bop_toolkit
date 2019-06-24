# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the error of 6D object pose estimates."""

import os
import time
import argparse

from bop_toolkit import config
from bop_toolkit import dataset_params
from bop_toolkit import inout
from bop_toolkit import misc
from bop_toolkit import pose_error
from bop_toolkit import renderer


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
  'renderer_type': 'python',  # Options: 'cpp', 'python'.

  # Names of files with results for which the errors will be calculated.
  # The files are assumed to be stored in folder config.eval_path.
  # See docs/bop_challenge_2019.md for a format description.
  'result_fnames': [
    'hodan-iros15-dv1-nopso_icbin-test.csv',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # Folder in which the calculated errors will be saved.
  'out_errors_dir': config.eval_path,

  # Template of path to the output file with calculated errors.
  'out_errors_tpath': os.path.join(
    '{out_errors_dir}', '{result_name}', '{error_sign}',
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
parser.add_argument('--result_fnames', default=','.join(p['result_fnames']),
                    help='Comma-separated names of files with results.')
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--out_errors_dir', default=p['out_errors_dir'])
parser.add_argument('--out_errors_tpath', default=p['out_errors_tpath'])
args = parser.parse_args()

p['result_fnames'] = args.result_fnames.split(',')
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

for result_fname in p['result_fnames']:
  misc.log('Processing: ' + result_fname)

  # Parse info about the method and the dataset from the folder name.
  result_name = os.path.splitext(os.path.basename(result_fname))[0]
  result_info = result_name.split('_')
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None
  split_type_str = ' - ' + split_type if split_type is not None else ''

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'eval'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

  # Load object models.
  models = {}
  if p['error_type'] in ['add', 'adi']:
    misc.log('Loading object models...')
    for obj_id in dp_model['obj_ids']:
      models[obj_id] = inout.load_ply(dp_model['model_tpath'].format(obj_id))

  # Initialize a renderer.
  ren = None
  if p['error_type'] in ['vsd', 'cou_mask_proj']:
    misc.log('Initializing renderer...')
    width, height = dp_split['im_size']
    ren = renderer.create_renderer(
      width, height, p['renderer_type'], mode='depth')
    for obj_id in dp_model['obj_ids']:
      ren.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))

  # Load pose estimates.
  misc.log('Loading pose estimates...')
  ests = inout.load_bop_results(
    os.path.join(config.results_path, result_fname))

  # Organize the pose estimates by scene, image and object.
  misc.log('Organizing pose estimates...')
  ests_org = {}
  for est in ests:
    ests_org.setdefault(
      est['scene_id'], {}).setdefault(
      est['im_id'], {}).setdefault(
      est['obj_id'], []).append(est)

  ests_counter = 0
  time_start = time.time()

  for scene_id, scene_ests in ests_org.items():

    # Load info and GT poses for the current scene.
    scene_camera = inout.load_scene_camera(
      dp_split['scene_camera_tpath'].format(scene_id=scene_id))
    scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(
      scene_id=scene_id))

    scene_errs = []
    depth_im = None
    for im_ind, (im_id, im_ests) in enumerate(scene_ests.items()):

      if im_ind % 10 == 0:
        misc.log(
          'Calculating error {} - method: {}, dataset: {}{}, scene: {}, '
          'im: {}'.format(
            p['error_type'], method, dataset, split_type_str, scene_id, im_ind))

      # Camera matrix.
      K = scene_camera[im_id]['cam_K']

      # Load the depth image if VSD is selected as the pose error function.
      if p['error_type'] == 'vsd':
        depth_path = dp_split['depth_tpath'].format(
          scene_id=scene_id, im_id=im_id)
        depth_im = inout.load_depth(depth_path)
        depth_im *= scene_camera[im_id]['depth_scale']  # Convert to [mm].

      for obj_id, obj_ests in im_ests.items():

        # Sort the estimates by score (in descending order).
        obj_ests_sorted = sorted(
          enumerate(obj_ests), key=lambda x: x[1]['score'], reverse=True)

        # Select the required number of top estimated poses.
        if p['n_top'] == 0:  # All estimates are considered.
          n_top_curr = None
        elif p['n_top'] == -1:  # Given by the number of GT poses.
          n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
          n_top_curr = n_gt
        else:
          n_top_curr = p['n_top']
        obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

        ests_counter += len(obj_ests_sorted)

        # Calculate error of each pose estimate w.r.t. all GT poses of the same
        # object class.
        for est_id, est in obj_ests_sorted:

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
              e = [pose_error.vsd(
                R_e, t_e, R_g, t_g, depth_im, K, p['vsd_delta'], p['vsd_tau'],
                ren, obj_id, p['vsd_cost'])]

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

  time_total = time.time() - time_start
  misc.log('{} estimates evaluated in {}s.'.format(ests_counter, time_total))

misc.log('Done.')
