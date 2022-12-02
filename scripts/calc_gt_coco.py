# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition

"""Calculates Instance Mask Annotations in Coco Format."""

import numpy as np
import os
import datetime
import json

from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib import dataset_params as dataset_p
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

def coco(par):
    dataset_params = dataset_p.DatasetParams(par)
    # PARAMETERS.
    ################################################################################
    p = {
    # See dataset_params.py for options.
    'dataset': par.name,

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': par.split,

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': par.split_type,

    # bbox type. Options: 'modal', 'amodal'.
    'bbox_type': 'amodal',

    # Folder containing the BOP datasets.
    'datasets_path': par.dataset_path,

    }
    ################################################################################

    datasets_path = p['datasets_path']
    dataset_name = p['dataset']
    split = p['dataset_split']
    split_type = p['dataset_split_type']
    bbox_type = p['bbox_type']


    
    dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
    dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

    complete_split = split
    if dp_split['split_type'] is not None:
        complete_split += '_' + dp_split['split_type']

    CATEGORIES = [{'id': obj_id, 'name':str(obj_id), 'supercategory': dataset_name} for obj_id in dp_model['obj_ids']]
    INFO = {
        "description": dataset_name + '_' + split,
        "url": "https://github.com/thodan/bop_toolkit",
        "version": "0.1.0",
        "year": datetime.date.today().year,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    scene_ids = dataset_params.get_present_scene_ids(dp_split)

    for scene_id in scene_ids:
        segmentation_id = 1

        coco_scene_output = {
            "info": INFO,
            "licenses": [],
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        # Load info about the GT poses (e.g. visibility) for the current scene.
        scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(scene_id=scene_id))
        scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id), keys_to_int=True)
        # Output coco path
        coco_gt_path = dp_split['scene_gt_coco_tpath'].format(scene_id=scene_id)
        if bbox_type == 'modal':
            coco_gt_path = coco_gt_path.replace('scene_gt_coco', 'scene_gt_coco_modal')
        misc.log('Calculating Coco Annotations - dataset: {} ({}, {}), scene: {}'.format(
            p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id))
        
        # Go through each view in scene_gt
        for scene_view, inst_list in scene_gt.items():
            im_id = int(scene_view)
            
            img_path = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
            relative_img_path = os.path.relpath(img_path, os.path.dirname(coco_gt_path))
            image_info = pycoco_utils.create_image_info(im_id, relative_img_path, dp_split['im_size'])
            coco_scene_output["images"].append(image_info)
            gt_info = scene_gt_info[scene_view]
            
            # Go through each instance in view
            for idx,inst in enumerate(inst_list): 
                category_info = inst['obj_id']
                visibility = gt_info[idx]['visib_fract']
                # Add ignore flag for objects smaller than 10% visible
                ignore_gt = visibility < 0.1
                mask_visib_p = dp_split['mask_visib_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=idx)
                mask_full_p = dp_split['mask_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=idx)
                
                binary_inst_mask_visib = inout.load_depth(mask_visib_p).astype(np.bool)
                if binary_inst_mask_visib.sum() < 1:
                    continue
                if bbox_type == 'amodal':
                    binary_inst_mask_full = inout.load_depth(mask_full_p).astype(np.bool)
                    if binary_inst_mask_full.sum() < 1:
                        continue
                    bounding_box = pycoco_utils.bbox_from_binary_mask(binary_inst_mask_full)
                elif bbox_type == 'modal':
                    bounding_box = pycoco_utils.bbox_from_binary_mask(binary_inst_mask_visib)
                else:
                    raise Exception('{} is not a valid bounding box type'.format(p['bbox_type']))

                annotation_info = pycoco_utils.create_annotation_info(
                    segmentation_id, im_id, category_info, binary_inst_mask_visib, bounding_box, tolerance=2, ignore=ignore_gt)

                if annotation_info is not None:
                    coco_scene_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

        with open(coco_gt_path, 'w') as output_json_file:
            json.dump(coco_scene_output, output_json_file)
