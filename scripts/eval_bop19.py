# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019."""

import os
import time
import argparse
import subprocess

from bop_toolkit import config
from bop_toolkit import inout
from bop_toolkit import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
  # Errors to calculate.
  'errors': [
    {
      'n_top': -1,
      'type': 'vsd',
      'vsd_delta': 15,
      'vsd_tau': 20,
      'correct_th': [[0.3]]
    },
    {
      'n_top': -1,
      'type': 'cou-mask-proj',
      'correct_th': [[0.3]]
    },
    {
      'n_top': -1,
      'type': 'ad',
      'correct_th': [[0.1]]
    },
  ],

  # Minimum visible surface fraction of a valid GT pose.
  'visib_gt_min': 0.1,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'python',  # Options: 'cpp', 'python'.

  # Names of files with results for which to calculate the errors (assumed to be
  # stored in folder config.eval_path). See docs/bop_challenge_2019.md for a
  # description of the format. Example results can be found at:
  # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
  'result_filenames': [
    '/path/to/csv/with/results',
  ],

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  'targets_filename': 'test_targets_bop19.yml',
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--visib_gt_min', default=p['visib_gt_min'])
parser.add_argument('--renderer_type', default=p['renderer_type'])
parser.add_argument('--result_filenames',
                    default=','.join(p['result_filenames']),
                    help='Comma-separated names of files with results.')
parser.add_argument('--targets_filename', default=p['targets_filename'])
args = parser.parse_args()

p['visib_gt_min'] = float(args.visib_gt_min)
p['renderer_type'] = str(args.renderer_type)
p['result_filenames'] = args.result_filenames.split(',')
p['targets_filename'] = str(args.targets_filename)

# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p['result_filenames']:

  misc.log('-----------')
  misc.log('EVALUATING: {}'.format(result_filename))
  misc.log('-----------')

  time_start = time.time()

  for error in p['errors']:

    # Calculate error of the pose estimates.
    calc_errors_cmd = [
      'python',
      os.path.join('scripts', 'eval_calc_errors.py'),
      '--n_top={}'.format(error['n_top']),
      '--error_type={}'.format(error['type']),
      '--result_filenames={}'.format(result_filename),
      '--renderer_type={}'.format(p['renderer_type']),
      '--targets_filename={}'.format(p['targets_filename']),
      '--skip_missing=1',
    ]
    if error['type'] == 'vsd':
      calc_errors_cmd += [
        '--vsd_delta={}'.format(error['vsd_delta']),
        '--vsd_tau={}'.format(error['vsd_tau'])
      ]

    misc.log('Running: ' + ' '.join(calc_errors_cmd))
    if subprocess.call(calc_errors_cmd) != 0:
      raise RuntimeError('Calculation of VSD failed.')

    # Path (relative to config.eval_path) to folder with calculated pose errors.
    if error['type'] == 'vsd':
      error_sign = misc.get_error_signature(
        error['type'], error['n_top'], vsd_delta=error['vsd_delta'],
        vsd_tau=error['vsd_tau'])
    else:
      error_sign = misc.get_error_signature(
        error['type'], error['n_top'])

    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    error_dir_path = os.path.join(result_name, error_sign)

    # Calculate performance scores.
    for correct_th in error['correct_th']:

      calc_scores_cmd = [
        'python',
        os.path.join('scripts', 'eval_calc_scores.py'),
        '--error_dir_paths={}'.format(error_dir_path),
        '--targets_filename={}'.format(p['targets_filename']),
        '--visib_gt_min={}'.format(p['visib_gt_min'])
      ]

      if error['type'] in ['add', 'adi']:
        calc_scores_cmd += ['--correct_th_fact_{}={}'.format(
          error['type'], ','.join(map(str, correct_th)))]
      else:
        calc_scores_cmd += ['--correct_th_{}={}'.format(
          error['type'], ','.join(map(str, correct_th)))]

      misc.log('Running: ' + ' '.join(calc_scores_cmd))
      if subprocess.call(calc_scores_cmd) != 0:
        raise RuntimeError('Calculation of scores failed.')

      # Path to file with calculated scores.
      if error['type'] in ['add', 'adi']:
        score_sign = misc.get_score_signature(
          error['type'], p['visib_gt_min'], correct_th_fact=correct_th)
      else:
        score_sign = misc.get_score_signature(
          error['type'], p['visib_gt_min'], correct_th=correct_th)

      scores_filename = 'scores_{}.yml'.format(score_sign)
      scores_path = os.path.join(
        config.eval_path, result_name, error_sign, scores_filename)

      # Load the scores.
      misc.log('Loading calculated scores from: {}'.format(scores_path))
      scores = inout.load_yaml(scores_path)

  time_total = time.time() - time_start
  misc.log('Evaluation of {} took {}s.'.format(result_filename, time_total))
