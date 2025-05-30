# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""
Create POSE result files from ground truth annotation and targets file.
"""

import os
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Dataset name. See dataset_params.py for options.
    "dataset": "xyzibd",
    # Dataset split. See dataset_params.py for options
    "split": "test",  
    # Dataset split type. See dataset_params.py for options
    "split_type": None,
    # Out perfect result file name 
    "results_name": 'gt-results',    
    # Predefined test targets 
    "targets_filename": "test_targets_bop24.json",
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Minimum visibility of the GT poses to include them in the output result file.
    "min_visib_gt": 0.0,  # bop24 uses 0.1, 0.0 -> all gts are stored
    # add gaussian noise to the GT translation if stdt > 0, in millimeters (e.g. 5)
    "stdt": 0.0,
    # add gaussian noise to the GT orientation if stdo > 0, in radians (e.g. 0.1)
    "stdo": 0.0,
    # RNG seed 
    "seed": 0,
    # keep only the n_first first results, None keeps all results 
    "n_first": None
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--results_name", default=p["results_name"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--split", default=p["split"])
parser.add_argument("--split_type", default=p["split_type"])
parser.add_argument("--min_visib_gt", default=p["min_visib_gt"])
parser.add_argument("--stdt", type=float, default=p["stdt"])
parser.add_argument("--stdo", type=float, default=p["stdo"])
parser.add_argument("--seed", type=int, default=p["seed"])
parser.add_argument("--n_first", default=p["n_first"])
args = parser.parse_args()

misc.log(f"Creating pose results from gt for {args.dataset}")
split_type = str(args.split_type) if args.split_type is not None else None
n_first = int(args.n_first) if args.n_first is not None else None

rng = np.random.default_rng(args.seed)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], args.dataset, args.split, split_type
)
if not os.path.exists(dp_split["base_path"]):
    misc.log(f'Dataset does not exist: {dp_split["base_path"]}')
    exit()

# Load and organize the estimation targets.
target_file_path = os.path.join(dp_split["base_path"], args.targets_filename)
targets = inout.load_json(target_file_path)
targets_org = misc.reorganize_targets(targets)

results = []

for scene_id in targets_org:
    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

    scene_gt_path = dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
    scene_gt_info_path = dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
    scene_gt = inout.load_scene_gt(scene_gt_path)
    scene_gt_info = inout.load_scene_gt(scene_gt_info_path)

    for im_id in targets_org[scene_id]:
        img_gt = scene_gt[im_id]
        img_gt_info = scene_gt_info[im_id]

        for obj_gt, obj_gt_info in zip(img_gt, img_gt_info):
            if obj_gt_info['visib_fract'] >= args.min_visib_gt:
                cam_R_m2c = obj_gt["cam_R_m2c"]
                cam_t_m2c = obj_gt["cam_t_m2c"]
                if args.stdo > 0.0:
                    # apply a local perturbation to the rotation matrix
                    rot_vec_pert = rng.normal(loc=0.0, scale=args.stdo*np.ones(3))
                    cam_R_m2c = cam_R_m2c @ R.from_rotvec(rot_vec_pert).as_matrix()
                if args.stdt > 0.0:
                    cam_t_m2c += rng.normal(loc=0.0, scale=args.stdt*np.ones(3)).reshape((3,1))
                result = {
                    "scene_id": int(scene_id),
                    "im_id": int(im_id),
                    "obj_id": int(obj_gt["obj_id"]),
                    "score": 1.0,
                    "R": cam_R_m2c,
                    "t": cam_t_m2c,
                    "time": -1.0,
                }
                results.append(result)


if n_first is not None:
    results = results[:n_first]


result_filename = f"{args.results_name}_{args.dataset}-{args.split}_pose.csv"
if args.stdt > 0.0 or args.stdo > 0.0:
    result_filename = f"{args.results_name}_{args.dataset}-{args.split}_stdt={args.stdt}_stdo={args.stdo}_pose.csv"
if not os.path.exists(args.results_path):
    misc.log(f"Creating dir {args.results_path}")
    os.mkdir(args.results_path)
results_path = os.path.join(args.results_path, result_filename)
inout.save_bop_results(results_path, results)
misc.log(f"Saved {results_path}")
