# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Main evaluation script."""

import os
import subprocess

from bop_toolkit import misc


result_filename = 'hodan-iros15-dv1-nopso_lm-test.csv'


# Errors to calculate.
errors = [
  {'n_top': -1, 'type': 'vsd', 'vsd_delta': 15, 'vsd_tau': 20},
  {'n_top': -1, 'type': 'adi'}
]

for error in errors:

  # Calculate error of the pose estimates.
  calc_errors_cmd = [
    'python',
    os.path.join('scripts', 'eval_calc_errors.py'),
    '--n_top={}'.format(error['n_top']),
    '--error_type={}'.format(error['type']),
    '--result_filenames={}'.format(result_filename),
    '--targets_filename=test_targets_bopc19.yml'
    '--renderer_type=cpp',
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
  error_sign = misc.get_error_signature(
    error['type'], error['n_top'], error['vsd_delta'], error['vsd_tau'])
  result_name = os.path.splitext(os.path.basename(result_filename))[0]
  error_dir_path = os.path.join(result_name, error_sign)

  # Calculate performance scores.
  calc_scores_cmd = [
    'python',
    os.path.join('scripts', 'eval_calc_scores.py'),
    '--n_top={}'.format(error['n_top']),
    '--error_type={}'.format(error['type']),
    '--vsd_delta={}'.format(error['vsd_delta']),
    '--vsd_tau={}'.format(error['vsd_tau']),
    '--error_dir_paths={}'.format(error_dir_path),
    '--visib_gt_min=0.1'
    '--visib_delta=15'
  ]

  if error['type'] in ['add', 'adi']:
    calc_scores_cmd += ['--error_th_fact_{}={}'.format(
      error['type'], error[''])]
  else:
    calc_scores_cmd += ['--error_th_{}={}'.format(
      error['type'], error[''])]
