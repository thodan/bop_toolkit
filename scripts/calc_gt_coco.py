# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition

"""Calculates Instance Mask Annotations in Coco Format."""

import os
import argparse
import datetime
import json

from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "xyzibd",
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "test",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Predefined test targets, either 'test_targets_bop19.json' or 'test_targets_bop24.json'. 
    "targets_filename": "test_targets_bop19.json",
    # Instead of using the predefined test targets, use all GT poses. 
    "use_all_gt": False,
    # bbox type. Options: 'modal', 'amodal'.
    "bbox_type": "amodal",
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--dataset_split", default=p["dataset_split"])
parser.add_argument("--dataset_split_type", default=p["dataset_split_type"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--use_all_gt", action="store_true", default=p["use_all_gt"])
parser.add_argument("--bbox_type", default=p["bbox_type"])
parser.add_argument("--datasets_path", default=p["datasets_path"])
args = parser.parse_args()

split_type = None if args.dataset_split_type is None else str(args.dataset_split_type)

dp_split = dataset_params.get_split_params(
    args.datasets_path, args.dataset, args.dataset_split, split_type=split_type
)
dp_model = dataset_params.get_model_params(args.datasets_path, args.dataset)

complete_split = args.dataset_split
if dp_split["split_type"] is not None:
    complete_split += "_" + dp_split["split_type"]

CATEGORIES = [
    {"id": obj_id, "name": str(obj_id), "supercategory": args.dataset}
    for obj_id in dp_model["obj_ids"]
]
INFO = {
    "description": args.dataset + "_" + args.dataset_split,
    "url": "https://github.com/thodan/bop_toolkit",
    "version": "0.1.0",
    "year": datetime.date.today().year,
    "contributor": "",
    "date_created": datetime.datetime.now(datetime.timezone.utc).isoformat(" "),
}

# Load and organize the estimation targets.
target_file_path = os.path.join(dp_split["base_path"], p["targets_filename"])
targets = inout.load_json(target_file_path)
targets_org = misc.reorganize_targets(targets)

for scene_id in dp_split["scene_ids"]:
    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)
    scene_modality = dataset_params.get_scene_sensor_or_modality(dp_split["eval_modality"], scene_id)
    scene_sensor = dataset_params.get_scene_sensor_or_modality(dp_split["eval_sensor"], scene_id)

    segmentation_id = 1

    coco_scene_output = {
        "info": INFO,
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))
    scene_gt_info = inout.load_json(
        dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id), keys_to_int=True
    )
    scene_camera = inout.load_scene_camera(
        dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
    )
    # Output coco path
    coco_gt_path = dp_split[tpath_keys["scene_gt_coco_tpath"]].format(scene_id=scene_id)
    if args.bbox_type == "modal":
        coco_gt_path = coco_gt_path.replace("scene_gt_coco", "scene_gt_coco_modal")
    misc.log(
        "Calculating Coco Annotations - dataset: {} ({}, {}), scene: {}".format(
            p["dataset"], p["dataset_split"], p["dataset_split_type"], scene_id
        )
    )

    # Go through each view in scene_gt
    for im_id, inst_list in scene_gt.items():
        # Skip if the image is not in the targets
        in_target = scene_id in targets_org and im_id in targets_org[scene_id]
        if not args.use_all_gt and not in_target:
            misc.log("Skip image {} in scene {}".format(im_id, scene_id))
            continue

        img_path = dp_split[tpath_keys["rgb_tpath"]].format(scene_id=scene_id, im_id=im_id)
        relative_img_path = os.path.relpath(img_path, os.path.dirname(coco_gt_path))
        im_size = dataset_params.get_im_size(dp_split, scene_modality, scene_sensor)
        image_info = pycoco_utils.create_image_info(
            im_id, relative_img_path, im_size
        )
        coco_scene_output["images"].append(image_info)
        gt_info = scene_gt_info[im_id]

        # Go through each instance in view
        for idx, inst in enumerate(inst_list):
            category_info = inst["obj_id"]
            visibility = gt_info[idx]["visib_fract"]
            # Add ignore flag for objects smaller than 10% visible
            ignore_gt = visibility < 0.1
            mask_visib_p = dp_split[tpath_keys["mask_visib_tpath"]].format(
                scene_id=scene_id, im_id=im_id, gt_id=idx
            )
            mask_full_p = dp_split[tpath_keys["mask_tpath"]].format(
                scene_id=scene_id, im_id=im_id, gt_id=idx
            )

            binary_inst_mask_visib = inout.load_depth(mask_visib_p).astype(bool)
            if binary_inst_mask_visib.sum() < 1:
                continue
            if args.bbox_type == "amodal":
                binary_inst_mask_full = inout.load_depth(mask_full_p).astype(bool)
                if binary_inst_mask_full.sum() < 1:
                    continue
                bounding_box = pycoco_utils.bbox_from_binary_mask(binary_inst_mask_full)
            elif args.bbox_type == "modal":
                bounding_box = pycoco_utils.bbox_from_binary_mask(
                    binary_inst_mask_visib
                )
            else:
                raise Exception(
                    "{} is not a valid bounding box type".format(args.bbox_type)
                )

            annotation_info = pycoco_utils.create_annotation_info(
                segmentation_id,
                im_id,
                category_info,
                binary_inst_mask_visib,
                bounding_box,
                tolerance=2,
                ignore=ignore_gt,
            )

            if annotation_info is not None:
                coco_scene_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1

    with open(coco_gt_path, "w") as output_json_file:
        json.dump(coco_scene_output, output_json_file)
        misc.log('Saved {}'.format(coco_gt_path))
