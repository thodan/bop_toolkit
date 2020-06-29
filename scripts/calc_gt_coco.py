# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition

"""Calculates Instance Mask Annotations in Coco Format."""

import math
import numpy as np
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
  'dataset': 'lmo',

  # Dataset split. Options: 'train', 'test'.
  'dataset_split': 'train',

  # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
  'dataset_split_type': 'pbr',

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

}
################################################################################


datasets_path = p['datasets_path']
dataset_name = p['dataset']
split = p['dataset_split']
split_type = p['dataset_split_type']

dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

complete_split = split
if dp_split['split_type'] is not None:
    complete_split += '_' + dp_split['split_type']

ROOT_DIR = os.path.join(dp_split['base_path'], complete_split, 'coco_annotations')
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

CATEGORIES = [{'id': obj_id, 'name':str(obj_id), 'supercategory': dataset_name} for obj_id in dp_model['obj_ids']]
INFO = {
    "description": dataset_name + '_' + split,
    "url": "https://github.com/thodan/bop_toolkit",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

for scene_id in dp_split['scene_ids']:
    image_id = 0
    segmentation_id = 1

    coco_scene_output = {
        "info": INFO,
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(scene_id=scene_id))

    misc.log('Calculating Coco Annotations - dataset: {} ({}, {}), scene: {}'.format(
          p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id))

    for scene_view, inst_list in scene_gt.items():
        im_id = int(scene_view)
        mask_paths = os.path.join(dp_split['base_path'], complete_split, '{:06d}/mask_visib'.format(scene_id))
        img_path = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
        relative_img_path = os.path.relpath(img_path, ROOT_DIR)
        
        image_info = pycoco_utils.create_image_info(image_id, relative_img_path, dp_split['im_size'])
        coco_scene_output["images"].append(image_info)
        
        for idx,inst in enumerate(inst_list): 
            category_info = inst['obj_id']

            mask_p = os.path.join(mask_paths, '{:06d}_{:06d}.png'.format(im_id, idx))
            binary_inst_mask = (inout.load_depth(mask_p)/255.).astype(np.bool)

            annotation_info = pycoco_utils.create_annotation_info(
                segmentation_id, image_id, category_info, binary_inst_mask, tolerance=2)

            if annotation_info is not None:
                coco_scene_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open('{}/coco_{}_{}_{:06d}.json'.format(ROOT_DIR, dataset_name, split, scene_id), 'w') as output_json_file:
        json.dump(coco_scene_output, output_json_file)


