"""
This script will read the "scene_gt_world.json" file and project the annotations from the world frame to each camera frame.

This script is supposed to be used after the annotations have been manually done in the annotation tool.
"""

import os
import glob
import json

import numpy as np
import open3d as o3d

# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/path/to/dataset',

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': 'train',

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': None,
}
################################################################################


def main():
    dataset_split_path = os.path.join(
        p['dataset_path'],
        p['dataset_split'] + '_' + p['dataset_split_type'] if p['dataset_split_type'] else p['dataset_split'])
    scenes_paths = glob.glob(dataset_split_path + '/*')

    for scene_path in scenes_paths:  # samples are not ordered
        print("Processing scene: ", scene_path)
        scene_camera_world_json = os.path.join(scene_path, 'scene_gt_world.json')
        scene_camera_json = os.path.join(scene_path, 'scene_camera.json')
        scene_gt_json = os.path.join(scene_path, 'scene_gt.json')
        with open(scene_camera_world_json) as j:
            scene_camera_world_data = json.load(j)
        with open(scene_camera_json) as j:
            scene_camera_data = json.load(j)

        objs_world_annos = scene_camera_world_data['w']

        scene_gt_data = {}

        for frame_idx, cam_info in scene_camera_data.items():
            cam_trans = np.array(cam_info['cam_t_w2c'])
            cam_rot = np.array(cam_info['cam_R_w2c']).reshape(3,3)
            H_camZ_V = np.eye(4)
            H_camZ_V[:3,:3] = cam_rot
            H_camZ_V[:3,3] = cam_trans

            frame_anno_list = []
            for obj_world_anno in objs_world_annos:
                world_trans = np.array(obj_world_anno['cam_t_m2c'])
                world_rot = np.array(obj_world_anno['cam_R_m2c']).reshape(3,3)
                H_V_obj = np.eye(4)
                H_V_obj[:3,:3] = world_rot
                H_V_obj[:3,3] = world_trans

                H_camZ_obj = H_camZ_V @ H_V_obj

                frame_anno_list.append({
                    'cam_t_m2c': H_camZ_obj[:3,3].flatten().tolist(),
                    'cam_R_m2c': H_camZ_obj[:3,:3].tolist(),
                    'obj_id': obj_world_anno['obj_id']
                })

            scene_gt_data[frame_idx] = frame_anno_list

        with open(scene_gt_json, 'w') as f:
            json.dump(scene_gt_data, f)


if __name__ == "__main__":
    main()
