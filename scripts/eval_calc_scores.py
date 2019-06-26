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
 - correct_th['vsd'] = 0.3

 [1] Hodan, Michel et al. BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.
"""

import os
import time
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
  'correct_th': {
    'vsd': [0.3],
    'cou-mask-proj': [0.5],
    'rete': [5.0, 5.0],  # [deg, cm].
    're': [5.0],  # [deg].
    'te': [5.0]  # [cm].
  },

  # Factor k; threshold of correctness = k * d, where d is the obj. diameter.
  'correct_th_fact': {
    'ad': [0.1],
    'add': [0.1],
    'adi': [0.1]
  },

  # Minimum visible surface fraction of a valid GT pose.
  'visib_gt_min': 0.1,

  # Paths (relative to config.eval_path) to folders with pose errors calculated
  # using eval_calc_errors.py.
  # Example: 'hodan-iros15_lm-test/error=vsd_ntop=1_delta=15_tau=20_cost=step'
  'error_dir_paths': [
    r'/path/to/calculated/errors',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  'targets_filename': 'test_targets_bop19.yml',

  # Template of path to the input file with calculated errors.
  'error_tpath': os.path.join(
    config.eval_path, '{error_dir_path}', 'errors_{scene_id:06d}.yml'),

  # Template of path to the output file with established matches and calculated
  # scores.
  'out_matches_tpath': os.path.join(
    config.eval_path, '{error_dir_path}', 'matches_{score_sign}.yml'),
  'out_scores_tpath': os.path.join(
    config.eval_path, '{error_dir_path}', 'scores_{score_sign}.yml'),
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Define the command line arguments.
for err_type in p['correct_th']:
  parser.add_argument(
    '--correct_th_' + err_type,
    default=','.join(map(str, p['correct_th'][err_type])))

for err_type in p['correct_th_fact']:
  parser.add_argument(
    '--correct_th_fact_' + err_type,
    default=','.join(map(str, p['correct_th_fact'][err_type])))

parser.add_argument('--visib_gt_min', default=p['visib_gt_min'])
parser.add_argument('--error_dir_paths', default=','.join(p['error_dir_paths']),
                    help='Comma-sep. paths to errors from eval_calc_errors.py.')
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--targets_filename', default=p['targets_filename'])
parser.add_argument('--error_tpath', default=p['error_tpath'])
parser.add_argument('--out_matches_tpath', default=p['out_matches_tpath'])
parser.add_argument('--out_scores_tpath', default=p['out_scores_tpath'])

# Process the command line arguments.
args = parser.parse_args()

for err_type in p['correct_th']:
  p['correct_th'][err_type] =\
    map(float, args.__dict__['correct_th_' + err_type].split(','))

for err_type in p['correct_th_fact']:
  p['correct_th_fact'][err_type] =\
    map(float, args.__dict__['correct_th_fact_' + err_type].split(','))

p['visib_gt_min'] = float(args.visib_gt_min)
p['error_dir_paths'] = args.error_dir_paths.split(',')
p['datasets_path'] = str(args.datasets_path)
p['targets_filename'] = str(args.targets_filename)
p['error_tpath'] = str(args.error_tpath)
p['out_matches_tpath'] = str(args.out_matches_tpath)
p['out_scores_tpath'] = str(args.out_scores_tpath)

misc.log('-----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('-----------')

# Calculation of the performance scores.
# ------------------------------------------------------------------------------
for error_dir_path in p['error_dir_paths']:
  misc.log('Processing: {}'.format(error_dir_path))

  time_start = time.time()

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
  if err_type in ['ad', 'add', 'adi']:
    score_sign = misc.get_score_signature(
      err_type, p['visib_gt_min'],
      correct_th_fact=p['correct_th_fact'][err_type])
  else:
    score_sign = misc.get_score_signature(
      err_type, p['visib_gt_min'],
      correct_th=p['correct_th'][err_type])

  misc.log('Calculating score - error: {}, method: {}, dataset: {}.'.format(
    err_type, method, dataset))

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'eval'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

  # Load the estimation targets to consider.
  targets = inout.load_yaml(
    os.path.join(dp_split['base_path'], p['targets_filename']))
  scene_im_ids = {}

  # Organize the targets by scene, image and object.
  misc.log('Organizing estimation targets...')
  targets_org = {}
  for target in targets:
    targets_org.setdefault(target['scene_id'], {}).setdefault(
      target['im_id'], {})[target['obj_id']] = target

  # Set threshold of correctness (might be different for each object).
  correct_obj_ths = {}
  if err_type in ['ad', 'add', 'adi']:
    # Relative to object diameter.
    models_info = inout.load_yaml(dp_model['models_info_path'])
    for obj_id in dp_model['obj_ids']:
      diameter = models_info[obj_id]['diameter']
      correct_obj_ths[obj_id] =\
          [t * diameter for t in p['correct_th_fact'][err_type]]
  else:
    # The same threshold for all objects.
    for obj_id in dp_model['obj_ids']:
      correct_obj_ths[obj_id] = p['correct_th'][err_type]

  # Go through the test scenes and match estimated poses to GT poses.
  # ----------------------------------------------------------------------------
  matches = []  # Stores info about the matching pose estimate for each GT pose.
  for scene_id, scene_targets in targets_org.items():

    # Load GT poses for the current scene.
    scene_gt = inout.load_scene_gt(
      dp_split['scene_gt_tpath'].format(scene_id=scene_id))

    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt_info = inout.load_yaml(
      dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))

    # Keep GT poses only for the selected targets.
    scene_gt_curr = {}
    scene_gt_info_curr = {}
    scene_gt_valid = {}
    for im_id, im_targets in scene_targets.items():
      scene_gt_curr[im_id] = scene_gt[im_id]

      # Determine which GT poses are valid.
      scene_gt_valid[im_id] = []
      im_gt_info = scene_gt_info[im_id]
      for gt_id, gt in enumerate(scene_gt[im_id]):
        is_target = gt['obj_id'] in im_targets.keys()
        is_visib = im_gt_info[gt_id]['visib_fract'] >= p['visib_gt_min']
        scene_gt_valid[im_id].append(is_target and is_visib)

    # Load pre-calculated errors of the pose estimates w.r.t. the GT poses.
    scene_errs_path = p['error_tpath'].format(
      error_dir_path=error_dir_path, scene_id=scene_id)
    scene_errs = inout.load_errors(scene_errs_path)

    # Match the estimated poses to the ground-truth poses.
    matches += pose_matching.match_poses_scene(
      scene_id, scene_gt_curr, scene_gt_valid, scene_errs, correct_obj_ths,
      n_top)

  # Calculate the performance scores.
  # ----------------------------------------------------------------------------
  # 6D object localization scores (SiSo if n_top = 1).
  scores = score.calc_localization_scores(
    dp_split['scene_ids'], dp_model['obj_ids'], matches, n_top)

  # Save scores.
  scores_path = p['out_scores_tpath'].format(
    error_dir_path=error_dir_path, score_sign=score_sign)
  inout.save_yaml(scores_path, scores)

  # Save matches.
  matches_path = p['out_matches_tpath'].format(
    error_dir_path=error_dir_path, score_sign=score_sign)
  inout.save_yaml(matches_path, matches)

  time_total = time.time() - time_start
  misc.log('Matching and score calculation took {}s.'.format(time_total))

misc.log('Done.')
