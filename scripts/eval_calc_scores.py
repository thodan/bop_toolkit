# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates performance scores for 6D object pose estimation tasks.

Errors of the pose estimates need to be pre-calculated with eval_calc_errors.py.

Currently supported tasks (see [1]):
- SiSo (a single instance of a single object)

For evaluation in the BOP paper [1], the following parameters were used:
 - n_top = 1
 - visib_gt_min = 0.1
 - error_type = 'vsd'
 - vsd_cost = 'step'
 - vsd_delta = 15
 - vsd_tau = 20
 - error_th['vsd'] = 0.3

 [1] Hodan, Michel et al. BOP: Benchmark for 6D Object Pose Estimation,
     ECCV 2018.
"""

import os
import argparse

from bop_toolkit import config
from bop_toolkit import dataset_params
from bop_toolkit import inout
from bop_toolkit import misc
from bop_toolkit import pose_matching
from bop_toolkit import score


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
  # Threshold of correctness for different pose error functions.
  'error_th': {
    'vsd': [0.3],
    'cou_mask_proj': [0.5],
    'rete': [5.0, 5.0],  # [deg, cm].
    're': [5.0],  # [deg].
    'te': [5.0]  # [cm].
  },

  # Factor k; threshold of correctness = k * d, where d is the obj. diameter.
  'error_th_fact': {
    'add': [0.1],
    'adi': [0.1]
  },

  'require_all_errors': True,  # Whether to break if some errors are missing.
  'visib_gt_min': 0.1,  # Minimum visible surface fraction of a valid GT pose.
  'visib_delta': 15,  # Tolerance for estimation of the visibility mask [mm].

  # Paths to pose errors (calculated using eval_calc_errors.py).
  'error_dir_paths': [
    r'/path/to/pose/errors',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # File with a list of estimation targets from which a list of images to use
  # will be extracted. The file is assumed to be stored in the dataset folder.
  # None = all images are considerer.
  'targets_filename': 'test_targets_bopc19.yml',

  # Template of path to the input file with calculated errors.
  'error_tpath': os.path.join('{error_path}', 'errors_{scene_id:06d}.yml'),

  # Template of path to the output file with established matches and calculated
  # scores.
  'out_matches_tpath': os.path.join('{error_path}', 'matches_{eval_sign}.yml'),
  'out_scores_tpath': os.path.join('{error_path}', 'scores_{eval_sign}.yml'),
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Define the command line arguments.
for err_type in p['error_th']:
  parser.add_argument(
    '--error_th_' + err_type,
    default=','.join(map(str, p['error_th'][err_type])))

for err_type in p['error_th_fact']:
  parser.add_argument(
    '--error_th_fact_' + err_type,
    default=','.join(map(str, p['error_th_fact'][err_type])))

parser.add_argument('--require_all_errors', default=p['require_all_errors'])
parser.add_argument('--visib_gt_min', default=p['visib_gt_min'])
parser.add_argument('--visib_delta', default=p['visib_delta'])
parser.add_argument('--error_dir_paths', default=','.join(p['error_dir_paths']),
                    help='Comma-sep. paths to errors from eval_calc_errors.py.')
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--targets_filename', default=p['targets_filename'])
parser.add_argument('--error_tpath', default=p['error_tpath'])
parser.add_argument('--out_matches_tpath', default=p['out_matches_tpath'])
parser.add_argument('--out_scores_tpath', default=p['out_scores_tpath'])

# Process the command line arguments.
args = parser.parse_args()

for err_type in p['error_th']:
  p['error_th'][err_type] =\
    map(float, args.__dict__['error_th_' + err_type].split(','))

for err_type in p['error_th_fact']:
  p['error_th_fact'][err_type] =\
    map(float, args.__dict__['error_th_fact_' + err_type].split(','))

p['require_all_errors'] = bool(args.require_all_errors)
p['visib_gt_min'] = float(args.visib_gt_min)
p['visib_delta'] = float(args.visib_delta)
p['error_dir_paths'] = args.error_dir_paths.split(',')
p['datasets_path'] = str(args.datasets_path)
p['targets_filename'] = str(args.targets_filename)
p['error_tpath'] = str(args.error_tpath)
p['out_matches_tpath'] = str(args.out_matches_tpath)
p['out_scores_tpath'] = str(args.out_scores_tpath)

misc.log('----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('----------')

# Calculation of the performance scores.
# ------------------------------------------------------------------------------
for error_dir_path in p['error_dir_paths']:

  # Parse info about the errors from the folder name.
  error_sign = os.path.basename(error_dir_path)
  err_type = str(error_sign.split('_')[0].split('=')[1])
  n_top = int(error_sign.split('_')[1].split('=')[1])
  result_info = os.path.basename(os.path.dirname(error_dir_path)).split('_')
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None

  # Evaluation signature.
  if err_type in ['add', 'adi']:
    eval_sign = 'thf=' + '-'.join((p['error_th_fact'][err_type]))
  else:
    eval_sign = 'th=' + '-'.join((map(str, p['error_th'][err_type])))
  eval_sign += '_min-visib=' + str(p['visib_gt_min'])

  misc.log('Calculating score - error: {}, method: {}, dataset: {}.'.format(
    err_type, method, dataset))

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'eval'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

  # Subset of images to consider.
  if p['targets_filename'] is not None:
    targets = inout.load_yaml(
      os.path.join(dp_split['base_path'], p['targets_filename']))
    scene_im_ids = {}
    for target in targets:
      scene_im_ids.setdefault(
        target['scene_id'], set()).add(target['im_id'])
  else:
    scene_im_ids = None

  # Set threshold of correctness (might be different for each object).
  error_obj_ths = {}
  if err_type in ['add', 'adi']:
    # Relative to object diameter.
    models_info = inout.load_yaml(dp_model['models_info_path'])
    for obj_id in dp_model['obj_ids']:
      diameter = models_info[obj_id]['diameter']
      error_obj_ths[obj_id] =\
          [t * diameter for t in p['error_th_fact'][err_type]]
  else:
    # The same threshold for all objects.
    for obj_id in dp_model['obj_ids']:
      error_obj_ths[obj_id] = p['error_th'][err_type]

  # Go through the test scenes and match estimated poses to GT poses.
  # ----------------------------------------------------------------------------
  matches = []  # Stores info about the matching pose estimate for each GT pose.
  for scene_id in dp_split['scene_ids']:

    # Load GT poses for the current scene.
    scene_gt = inout.load_scene_gt(
      dp_split['scene_gt_tpath'].format(scene_id=scene_id))

    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt_info = inout.load_yaml(
      dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))

    # Keep only the GT poses and their stats for the selected images.
    if scene_im_ids is not None:
      im_ids = scene_im_ids[scene_id]
      scene_gt = {im_id: scene_gt[im_id] for im_id in im_ids}
      scene_gt_info = {im_id: scene_gt_info[im_id] for im_id in im_ids}

    # Load pre-calculated errors of the pose estimates w.r.t. the GT poses.
    scene_errs_path = p['error_tpath'].format(
      error_path=error_dir_path, scene_id=scene_id)

    scene_errs = None
    if os.path.isfile(scene_errs_path):
      scene_errs = inout.load_errors(scene_errs_path)

    elif p['require_all_errors']:
      raise IOError('{} is missing, but errors for all scenes are required'
                    ' (require_all_errors = True).'.format(scene_errs_path))

    # Match the estimated poses to the ground-truth poses.
    matches += pose_matching.match_poses_scene(
      scene_id, scene_gt, scene_gt_info, scene_errs, p['visib_gt_min'],
      error_obj_ths, n_top)

  # Calculate the performance scores.
  # ----------------------------------------------------------------------------
  # 6D object localization scores (SiSo if n_top = 1).
  scores = score.calc_localization_scores(
    dp_split['scene_ids'], dp_model['obj_ids'], matches, n_top)

  # Save scores.
  scores_path = p['out_scores_tpath'].format(
    error_path=error_dir_path, eval_sign=eval_sign)
  inout.save_yaml(scores_path, scores)

  # Save matches.
  matches_path = p['out_matches_tpath'].format(
    error_path=error_dir_path, eval_sign=eval_sign)
  inout.save_yaml(matches_path, matches)

misc.log('Done.')
