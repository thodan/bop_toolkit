# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019/2020."""

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
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [
        "/path/to/csv/with/results",
    ],
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_filenames",
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
args = parser.parse_args()

p["result_filenames"] = args.result_filenames.split(",")


if __name__ == "__main__":
    for result_filename in p["result_filenames"]:
        result_path = os.path.join(config.results_path, result_filename)
        check_passed, check_msg = inout.check_bop_results(result_path, version="bop19")

        misc.log("Check msg: {}".format(check_msg))
