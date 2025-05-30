# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Generating estimation from GT for debugging/unit tests purposes."""

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from tqdm import tqdm
# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "hot3d", # Dataset name.
    # Only for hot3d dataset, whether counting only keyframes. 
    "only_keyframes": True,
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "test",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Minimum visibility of the GT poses to include them in the output.
    "min_visib_gt": 0.1,
}
################################################################################

datasets_path = p["datasets_path"]
dataset_name = p["dataset"]
split = p["dataset_split"]
split_type = p["dataset_split_type"]
min_visib_gt = p["min_visib_gt"]
print(f"Minimum visibility of the GT poses: {min_visib_gt}")

dp_split = dataset_params.get_split_params(
    datasets_path, dataset_name, split, split_type=split_type
)
num_scenes = dp_split["scene_ids"]
print(f"There are {len(num_scenes)} scenes in the dataset.")
num_images = 0
num_instances = 0

for scene_id in tqdm(dp_split["scene_ids"]):
    if dp_split["eval_modality"] is None:
        # Load info about the GT poses (e.g. visibility) for the current scene.
        scene_gt_info = inout.load_json(
            dp_split["scene_gt_info_tpath"].format(scene_id=scene_id), keys_to_int=True
        )
    else: # hot3d
        eval_modality = dp_split["eval_modality"](scene_id)
        # Load info about the GT poses (e.g. visibility) for the current scene.
        scene_gt_info_path = dp_split[f"scene_gt_info_{eval_modality}_tpath"].format(scene_id=scene_id)

        # Few scenes of HOT3D dataset are missing
        try:
            scene_gt_info = inout.load_json(
                scene_gt_info_path, keys_to_int=True
            )
        except FileNotFoundError:
            print(f"File not found: {scene_gt_info_path}")

    for im_id in scene_gt_info:
        # In case of HOT3D and only_keyframes, we only consider keyframes for counting instances
        if dataset_name == "hot3d" and p["only_keyframes"] and (im_id not in [14, 29, 44, 59, 74, 89, 104, 119, 134, 149]):
            continue
        num_images += 1
        im_gt_info = scene_gt_info[im_id]
        for gt_info in im_gt_info:
            if gt_info["visib_fract"] >= min_visib_gt:
                num_instances += 1

print(f"Total number of instances: {num_instances}")
print(f"Total number of images: {num_images}")