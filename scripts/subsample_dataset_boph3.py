# Author: Van Nguyen NGUYEN (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Script to subsample test set from the original BOP H3 dataset."""
# use: python scripts/subsample_dataset_boph3.py --dataset_name hot3d

import os
import argparse
from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import dataset_params
from tqdm import tqdm

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]

# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Dataset split. 
    "split": "test",
    "split_type": None,
    # Save file name with a list of targets to consider
    "targets_filename": "test_targets_bop24.json",
    # Min visib_fraction to consider the GT poses.
    "min_visib_gt": 0.1,
}
# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Dataset name (HOT3D, HANDAL).",
)
args = parser.parse_args()
p["dataset_name"] = str(args.dataset_name)
assert p["dataset_name"] in ["hot3d", "handal"], "Invalid dataset name."

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset_name"], p["split"], p["split_type"]
)

def check_scene_gt(scene_gt, scene_gt_info, im_ids):
    num_visib_instances = 0
    for im_id in im_ids:
        if f"{im_id}" not in scene_gt:
            return False, 0
        if f"{im_id}" not in scene_gt_info:
            return False, 0
        for gt_info in scene_gt_info[f"{im_id}"]:
            if gt_info["visib_fract"] >= p["min_visib_gt"]:
                num_visib_instances += 1
    return True, num_visib_instances

total_num_instances = 0
missing_scenes = []
# formatting test_list following BOP format: im_id, inst_count, obj_id, scene_id
test_list = []
for scene_id in tqdm(dp_split["scene_ids"]):
    if dp_split["eval_modality"] is None:
        modality = ""
        prefix = ""
    else:
        modality = dp_split["eval_modality"](scene_id)
        prefix = f"_{modality}"
    scene_gt_path = os.path.join(dp_split["base_path"], dp_split["split"], f"{scene_id:06d}", f"scene_gt{prefix}.json")
    scene_gt_info_path = os.path.join(dp_split["base_path"], dp_split["split"], f"{scene_id:06d}", f"scene_gt_info{prefix}.json")
    if not os.path.exists(scene_gt_path) or not os.path.exists(scene_gt_info_path):
        print("Missing scene_gt or scene_gt_info at {}".format(scene_gt_path))
        missing_scenes.append(scene_id)
        continue
    scene_gt = inout.load_json(scene_gt_path)
    scene_gt_info = inout.load_json(scene_gt_info_path)
    if p["dataset_name"] == "hot3d":
        selected_im_ids = [14, 29, 44, 59, 74, 89, 104, 119, 134, 149]
        gt_available, num_instances = check_scene_gt(scene_gt, scene_gt_info, selected_im_ids)
        assert gt_available, "Missing image ids in the scene_gt or scene_gt_info."
    elif p["dataset_name"] == "handal":
        avail_im_ids = sorted(list(scene_gt.keys()))
        # select with frame_rate = 0.125, i.e. every 8 frames
        selected_im_ids = [im_id for idx, im_id in enumerate(avail_im_ids) if idx % 8 == 0]
        _, num_instances = check_scene_gt(scene_gt, scene_gt_info, selected_im_ids)
    else:
        raise ValueError("Invalid dataset name.")
    total_num_instances += num_instances
    for im_id in selected_im_ids:
        test_list.append({"scene_id": scene_id, "im_id": int(im_id)})

num_total_scenes = len(dp_split["scene_ids"])
print(f"Missing {len(missing_scenes)}/{num_total_scenes} scenes")

print(f"Total number of test images: {len(test_list)}, total number of instances: {total_num_instances}")
out_path = os.path.join(dp_split["base_path"], p["targets_filename"])
inout.save_json(out_path, test_list)
print("Saved at {}".format(out_path))
