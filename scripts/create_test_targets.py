# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""
Create targets file for a given dataset.
"""

import os
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Test targets file produced by this script 
    "test_targets_filename": "test_targets_bop24.json",    
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    "dataset": "hot3d",  
    "split": "trainariasubsample",  
    "split_type": None,  
    # Select images from start to end every every_n_img
    "start": 14,
    "end": 150,
    "every_n_img": 15,
    # by default, we consider only objects that are at least 10% visible
    "visib_gt_min": 0.1,
    # detection targets: scene_id,im_id
    # localization targets: scene_id,im_id,obj_id,inst_count
    "eval_mode": "detection",  # Options: 'localization', 'detection'.
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--test_targets_filename", default=p["test_targets_filename"])
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--split", default=p["split"])
parser.add_argument("--split_type", default=p["split_type"])
parser.add_argument("--start", default=p["start"])
parser.add_argument("--end", default=p["end"])
parser.add_argument("--every_n_img", default=p["every_n_img"])
parser.add_argument("--visib_gt_min", default=p["visib_gt_min"])
parser.add_argument("--eval_mode", default=p["eval_mode"])
args = parser.parse_args()

split_type = str(args.split_type) if args.split_type is not None else None
every_n_img = int(args.every_n_img)
start = int(args.start)
end = int(args.end)
visib_gt_min = float(args.visib_gt_min)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], args.dataset, args.split, split_type
)

targets = []
for scene_id in dp_split["scene_ids"]:
    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], scene_id)
    scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))
    scene_gt_info = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id))
    selected_im_ids = list(scene_gt.keys())[start:end:every_n_img]
    for im_id in selected_im_ids:
        if args.eval_mode == "localization":
            im_targets = inout.get_im_targets(scene_gt[im_id], scene_gt_info[im_id], visib_gt_min, args.eval_mode)
            im_targets = [{"scene_id": scene_id, "im_id": im_id, "obj_id": obj_id, "inst_count": v["inst_count"]} 
                          for obj_id, v in im_targets.items()]
            targets += im_targets
        elif args.eval_mode == "detection":
            targets += [{"scene_id": scene_id, "im_id": im_id}]
        else:
            raise ValueError("{} eval_mode not supported".format(args.eval_mode))

targets_path = os.path.join(p["datasets_path"], args.dataset, args.test_targets_filename)
inout.save_json(targets_path, targets)
print('Saved ', targets_path)
