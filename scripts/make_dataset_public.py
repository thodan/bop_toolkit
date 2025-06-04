# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Generating estimation from GT for debugging/unit tests purposes."""

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
import os
import zipfile
from tqdm import tqdm
# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "handal",
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "test",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Minimum visibility of the GT poses to include them in the output.
    "min_visib_gt": 0.0,
    # Whether only for BOP test split.
    "is_bop_test_split": True,
    "targets_filename": "test_targets_bop24.json",
}
################################################################################

datasets_path = p["datasets_path"]
dataset_name = p["dataset"]
split = p["dataset_split"]
split_type = p["dataset_split_type"]
min_visib_gt = p["min_visib_gt"]

dp_split = dataset_params.get_split_params(
    datasets_path, dataset_name, split, split_type=split_type
)

# Load the estimation targets to consider.
targets = inout.load_json(
    os.path.join(dp_split["base_path"], p["targets_filename"])
)
targets_org = {}
for target in targets:
    targets_org.setdefault(target["scene_id"], {})[target["im_id"]] = target

output_path = os.path.join(
    datasets_path, f"{dataset_name}_{split}_public.zip"
)

num_selected_images = 0
num_ignored_images = 0

with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for scene_id in tqdm(dp_split["scene_ids"]):
        if p["is_bop_test_split"]:
            if scene_id not in targets_org:
                print(f"Scene {scene_id} not in the targets.")
                continue
            else:
                scene_targets = targets_org[scene_id]
        scene_dir = os.path.join(dp_split["split_path"], f"{scene_id:06d}")

        # Add scene_camera.json
        scene_camera_full_path = os.path.join(scene_dir, "scene_camera.json")
        relative_path = os.path.join(f"{split}", f"{scene_id:06d}", "scene_camera.json")
        zipf.write(scene_camera_full_path, relative_path)
        
        # RGB images
        scene_rgb_dir = os.path.join(scene_dir, "rgb")
        for root, dirs, files in os.walk(scene_rgb_dir):
            for file in tqdm(files):
                if p["is_bop_test_split"]:
                    im_id = int(file.split(".")[0])
                    if im_id not in scene_targets:
                        num_ignored_images += 1
                        continue
                # Create a full path for the file
                full_path = os.path.join(root, file)
                # Add file to the zip file, keeping its relative structure
                relative_path = os.path.join(f"{split}", f"{scene_id:06d}", "rgb", file)
                zipf.write(full_path, relative_path)
                num_selected_images += 1
                
print(f"Number of selected images: {num_selected_images}")
print(f"Number of ignored images: {num_ignored_images}")
print(f"Output written to: {output_path}")