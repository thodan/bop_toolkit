# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates performance scores for the 6D localization task.

For evaluation presented in the BOP paper (ECCV'18), the following parameters
were used:
 - n_top = 1
 - visib_gt_min = 0.1
 - error_type = 'vsd'
 - vsd_cost = 'step'
 - vsd_delta = 15
 - vsd_tau = 20
 - error_th['vsd'] = 0.3
"""

import os
from os.path import join
import argparse

from bop_toolkit import dataset_params, inout, misc, pose_matching, score


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
  # Threshold of correctness.
  'error_th': {
    'vsd': [0.3],
    'cou_mask_proj': [0.5],
    'rete': [5.0, 5.0],  # [deg,cm].
    're': [5.0],  # [deg].
    'te': [5.0]  # [cm].
  },

  # Factor k; threshold of correctness = k * d, where d is the obj. diameter.
  'error_th_fact': {
    'add': [0.1],
    'adi': [0.1]
  },

  'require_all_errors': True,  # Whether to break if some errors are missing.
  'visib_gt_min': 0.1,  # Minimum visible surface fraction of valid GT pose.
  'visib_delta': 15,  # [mm].

  # Paths to pose errors (calculated using calc_errors.py).
  'error_dir_paths': [
    r'C:\Users\tomho\th_data\cmp\projects\bop\test_data\eval\hodan-iros15-dv1-nopso_lm\error=re_ntop=1',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': r'C:\Users\tomho\th_data\msr\projects\pose6d\pose6d_scratch\datasets\bop',

  # Name of file with a list of image ID's to use for the evaluation. The file
  # is assumed to be stored in the dataset folder. None = all images are used
  # for the evaluation.
  'im_subset_filename': 'test_set_v1.yml',

  # Template of path to the input file with calculated errors.
  'error_tpath': join('{error_path}', 'errors_{scene_id:06d}.yml'),

  # Template of path to the output file with established matches and calculated
  # scores.
  'out_matches_tpath': join('{error_path}', 'matches_{eval_sign}.yml'),
  'out_scores_tpath': join('{error_path}', 'scores_{eval_sign}.yml'),
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
                    help='Comma-separated paths to errors from calc_errors.py.')
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--im_subset_filename', default=p['im_subset_filename'])
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
p['im_subset_filename'] = str(args.im_subset_filename)
p['error_tpath'] = str(args.error_tpath)
p['out_matches_tpath'] = str(args.out_matches_tpath)
p['out_scores_tpath'] = str(args.out_scores_tpath)

misc.log('----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('----------')


# Evaluation.
# ------------------------------------------------------------------------------
for error_dir_path in p['error_dir_paths']:

  # Parse info about the errors from the folder name.
  error_sign = os.path.basename(error_dir_path)
  err_type = error_sign.split('_')[0].split('=')[1]
  n_top = int(error_sign.split('_')[1].split('=')[1])
  result_info = os.path.basename(os.path.dirname(error_dir_path)).split('_')
  method = result_info[0]
  dataset = result_info[1]
  test_type = result_info[2] if len(result_info) > 3 else None

  # Evaluation signature.
  if err_type in ['add', 'adi']:
    eval_sign = 'thf=' + '-'.join((p['error_th_fact'][err_type]))
  else:
    eval_sign = 'th=' + '-'.join((map(str, p['error_th'][err_type])))
  eval_sign += '_min-visib=' + str(p['visib_gt_min'])

  misc.log('Evaluating - error: {}, method: {}, dataset: {}.'.format(
    err_type, method, dataset))

  # Load dataset parameters.
  dp = dataset_params.get_dataset_params(
    p['datasets_path'], dataset, test_type=test_type)
  obj_ids = dp['obj_ids']
  scene_ids = dp['scene_ids']

  # Subset of images to consider.
  if p['im_subset_filename'] is not None:
    im_subset_path = join(dp['base_path'], p['im_subset_filename'])
    im_ids_sets = inout.load_yaml(im_subset_path)
  else:
    im_ids_sets = None

  # Set threshold of correctness (might be different for each object).
  error_ths = {}
  if err_type in ['add', 'adi']:
    # Relative to object diameter.
    models_info = inout.load_yaml(dp['models_info_path'])
    for obj_id in obj_ids:
      diameter = models_info[obj_id]['diameter']
      error_ths[obj_id] = [t * diameter for t in p['error_th_fact'][err_type]]
  else:
    # The same threshold for all objects.
    for obj_id in obj_ids:
      error_ths[obj_id] = p['error_th'][err_type]

  # Go through the test scenes and match estimated poses to GT poses.
  # ----------------------------------------------------------------------------
  matches = []  # Stores info about the matching pose estimate for each GT.
  for scene_id in scene_ids:

    # Load GT poses.
    gts = inout.load_gt(dp['test_gt_tpath'].format(scene_id=scene_id))

    # Load statistics (e.g. visibility fraction) of the GT poses.
    gt_stats_path = dp['test_gt_stats_tpath'].format(
      scene_id=scene_id, delta=int(p['visib_delta']))
    gt_stats = inout.load_yaml(gt_stats_path)

    # Keep only the GT poses and their stats for the selected images.
    if im_ids_sets is not None:
      im_ids = im_ids_sets[scene_id]
      gts = {im_id: gts[im_id] for im_id in im_ids}
      gt_stats = {im_id: gt_stats[im_id] for im_id in im_ids}

    # Load pre-calculated errors of the pose estimates.
    scene_errs_path = p['error_tpath'].format(
      error_path=error_dir_path, scene_id=scene_id)

    if os.path.isfile(scene_errs_path):
      errs = inout.load_errors(scene_errs_path)

      # Matching of estimated poses to the ground-truth poses.
      matches += pose_matching.batch_match_poses(
        gts, gt_stats, errs, scene_id, p['visib_gt_min'], error_ths, n_top)

    elif p['require_all_errors']:
      raise IOError(
        '{} is missing, but errors for all scenes are required'
        ' (require_all_results = True).'.format(scene_errs_path)
      )

  # Calculate the performance scores.
  # ----------------------------------------------------------------------------
  scores = score.calc_siso_scores(scene_ids, obj_ids, matches, n_top)

  # Save scores.
  scores_path = p['out_scores_tpath'].format(
    error_path=error_dir_path, eval_sign=eval_sign)
  inout.save_yaml(scores_path, scores)

  # Save matches.
  matches_path = p['out_matches_tpath'].format(
    error_path=error_dir_path, eval_sign=eval_sign)
  inout.save_yaml(matches_path, matches)

misc.log('Done.')
