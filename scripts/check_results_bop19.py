# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019."""

import os
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
  # Names of files with results for which to calculate the errors (assumed to be
  # stored in folder config.eval_path). See docs/bop_challenge_2019.md for a
  # description of the format. Example results can be found at:
  # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
  'result_filenames': [
    '/path/to/csv/with/results',
  ],
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--result_filenames',
                    default=','.join(p['result_filenames']),
                    help='Comma-separated names of files with results.')
args = parser.parse_args()

p['result_filenames'] = args.result_filenames.split(',')

# Checking.
# ------------------------------------------------------------------------------
check_passed = True
for result_filename in p['result_filenames']:
  try:
    inout.load_bop_results(os.path.join(config.results_path, result_filename))
  except Exception as e:
    check_passed = False
    misc.log('ERROR when loading file {}:\n{}'.format(
      result_filename, e))

if check_passed:
  misc.log('Check passed.')
else:
  misc.log('Check failed.')
