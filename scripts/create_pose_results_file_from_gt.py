# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""
Create POSE result files from ground truth annotation and targets file.
"""

import os
import argparse

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
args = parser.parse_args()


p["results_path"] = str(args.results_path)
p["results_name"] = str(args.results_name)
p["targets_filename"] = str(args.targets_filename)
p["dataset"] = str(args.dataset)
p["split"] = str(args.split)
p["split_type"] = str(args.split_type) if args.split_type is not None else None
p["min_visib_gt"] = float(args.min_visib_gt)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["split"], p["split_type"]
)

if not os.path.exists(dp_split["base_path"]):
    misc.log(f'{dp_split["base_path"]} does not exist')

targets_path = os.path.join(p["datasets_path"], p["dataset"], p["targets_filename"])
targets = inout.load_json(targets_path)

# Load the estimation targets.
targets = inout.load_json(
    os.path.join(dp_split["base_path"], p["targets_filename"])
)

# Organize the targets by scene and image.
misc.log("Organizing estimation targets...")
targets_org = {}
for target in targets:
    targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})

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
            if obj_gt_info['visib_fract'] >= p["min_visib_gt"]:
                result = {
                    "scene_id": int(scene_id),
                    "im_id": int(im_id),
                    "obj_id": int(obj_gt["obj_id"]),
                    "score": 1.0,
                    "R": obj_gt["cam_R_m2c"],
                    "t": obj_gt["cam_t_m2c"],
                    "time": -1.0,
                }
                results.append(result)

result_filename = f"{p['results_name']}_{p['dataset']}-{p['split']}_pose.csv"
if not os.path.exists(p["results_path"]):
    misc.log(f"Creating dir {p['results_path']}")
    os.mkdir(p["results_path"])
results_path = os.path.join(p["results_path"], result_filename)
inout.save_bop_results(results_path, results)
misc.log(f"Saved {results_path}")
