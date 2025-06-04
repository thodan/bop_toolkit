# Author: Van Nguyen NGUYEN (van-nguyen.nguyen@enpc.fr)
# Imagine Team, ENPC, France

"""Cropping images and updating the scene_camera.json files accordingly."""

import os
import numpy as np
from tqdm import tqdm

from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import config


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "hope",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test_extension", # the output will be test_extension_cropped
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
}
################################################################################

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
)
dataset_save_dir = os.path.join(dp_split["base_path"], f"{p['dataset_split']}_cropped")
os.makedirs(dataset_save_dir, exist_ok=True)

# List of considered scenes.
scene_ids_curr = dp_split["scene_ids"]

# Create a renderer.
target_width, target_height = dp_split["im_size"]
misc.log(f"Target image size: {target_width}x{target_height}")

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:
    scene_save_dir = os.path.join(dataset_save_dir, f"{scene_id:06d}")
    for folder_name in ["rgb", "depth", "mask", "mask_visib"]:
        os.makedirs(os.path.join(scene_save_dir, folder_name), exist_ok=True)
    # we only change the scene_camera.json and scene_gt_info.json, not scene_gt.json
    scene_gt_path = dp_split["scene_gt_tpath"].format(scene_id=scene_id)
    new_scene_gt_path = scene_gt_path.replace(
        p["dataset_split"], p["dataset_split"] + "_cropped"
    )
    os.system(f"cp {scene_gt_path} {new_scene_gt_path}")

    # Load scene info and ground-truth poses.
    scene_camera_path = dp_split["scene_camera_tpath"].format(scene_id=scene_id)
    scene_camera = inout.load_scene_camera(scene_camera_path)
    scene_gt = inout.load_scene_gt(scene_gt_path)
    scene_gt_info = inout.load_json(
        dp_split["scene_gt_info_tpath"].format(scene_id=scene_id)
    )
    # List of considered images.
    im_ids = sorted(scene_gt.keys())

    # Render the object models in the ground-truth poses in the selected images.
    for im_counter, im_id in tqdm(enumerate(im_ids)):
        if im_counter % 10 == 0:
            misc.log("scene: {}, im: {}/{}".format(scene_id, im_counter, len(im_ids)))

        # Load the color and depth images and prepare images for rendering.
        if "rgb" in dp_split["im_modalities"] or p["dataset_split_type"] == "pbr":
            rgb_path = dp_split["rgb_tpath"].format(scene_id=scene_id, im_id=im_id)
        elif "gray" in dp_split["im_modalities"]:
            rgb_path = dp_split["gray_tpath"].format(scene_id=scene_id, im_id=im_id)
        else:
            raise ValueError("RGB nor gray images are available.")
        depth_path = dp_split["depth_tpath"].format(scene_id=scene_id, im_id=im_id)

        rgb = inout.load_im(rgb_path)
        depth = inout.load_depth(depth_path)

        # Crop the images rgb-d
        assert rgb.shape[:2] == depth.shape[:2]
        init_height, init_width = rgb.shape[:2]
        delta_w = (init_width - target_width) // 2
        delta_h = (init_height - target_height) // 2
        rgb = rgb[delta_h : target_height + delta_h, delta_w : target_width + delta_w]
        depth = depth[
            delta_h : target_height + delta_h, delta_w : target_width + delta_w
        ]
        cropped_rgb_path = rgb_path.replace(
            p["dataset_split"], p["dataset_split"] + "_cropped"
        )
        cropped_depth_path = depth_path.replace(
            p["dataset_split"], p["dataset_split"] + "_cropped"
        )

        inout.save_im(cropped_rgb_path, rgb)
        inout.save_depth(cropped_depth_path, depth)

        # Crop the masks
        for gt_id, gt in enumerate(scene_gt[im_id]):
            for mask_name in ["mask_tpath", "mask_visib_tpath"]:
                mask_path = dp_split[mask_name].format(
                    scene_id=scene_id, im_id=im_id, gt_id=gt_id
                )
                mask = inout.load_im(mask_path)
                assert mask.shape[:2] == (init_height, init_width)
                mask = mask[
                    delta_h : target_height + delta_h, delta_w : target_width + delta_w
                ]
                new_mask_path = mask_path.replace(
                    p["dataset_split"], p["dataset_split"] + "_cropped"
                )
                inout.save_im(new_mask_path, mask)

        # Update scene_camera.json
        K = scene_camera[im_id]["cam_K"]
        K[0, 2] -= delta_w
        K[1, 2] -= delta_h
        for k in scene_camera[im_id].keys():
            if isinstance(scene_camera[im_id][k], np.ndarray):
                scene_camera[im_id][k] = scene_camera[im_id][k].tolist()
    new_scene_camera_path = scene_camera_path.replace(
        p["dataset_split"], p["dataset_split"] + "_cropped"
    )
    inout.save_json(new_scene_camera_path, scene_camera)

    # double check that no image is missing
    for folder_name in ["rgb", "depth", "mask", "mask_visib"]:
        source_dir = os.path.join(
            dp_split["base_path"], p["dataset_split"], f"{scene_id:06d}", folder_name
        )
        target_dir = os.path.join(scene_save_dir, folder_name)
        source_files = len(os.listdir(source_dir))
        target_files = len(os.listdir(target_dir))
        if source_files != target_files:
            print(f"Error: {target_files} files in instead of {source_files}")
            print(f"Check the images in {target_dir}")

misc.log("Done.")
