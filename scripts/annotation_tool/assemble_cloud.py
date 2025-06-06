"""
This script will assemble a point cloud of consisting of all frames in each sample of the dataset.
This is required for annotating datasets that are captured in a sequence of frames.
All frames in a sample are assembled into on big point cloud in the first frame.

To be able to run this script scene_camera.json must include "cam_R_w2c" and "cam_t_w2c".
The script will save the assembled point cloud as assembled_cloud_world.pcd in the world_frame in each sample folder.

All units in this script are mm as in the BOP standard. The assembled point cloud is also saved in mm.
"""

import open3d as o3d
import copy
import numpy as np
import json

import os
import glob
from tqdm import tqdm


# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/path/to/dataset',

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': 'train',

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': None,

    # list of scene ids to process, if None all scenes in the dataset split are processed
    'scene_ids': None,

    # downsample the assembled point cloud to this voxel size in mm
    'voxel_size': 3,  # in mm

    'show_assembled_cloud': True
}
################################################################################


# Convert point clouds from legacy to tensors
def cloud_gpu(cloud):  # TODO use the function to get dynamic target
    cloud_copy = copy.deepcopy(cloud)
    cloud_t = o3d.t.geometry.PointCloud.from_legacy(cloud_copy)
    cloud_t.point["colors"] = cloud_t.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
    cloud_t = cloud_t.cuda(0)
    return cloud_t


def main():
    dataset_split_path = os.path.join(
        p['dataset_path'],
        p['dataset_split'] + '_' + p['dataset_split_type'] if p['dataset_split_type'] else p['dataset_split'])

    # if scene_ids is None, get all scene ids from the dataset split
    if p['scene_ids'] is None:
        scenes_paths = glob.glob(dataset_split_path + '/*')
        p['scene_ids'] = [int(os.path.basename(scene_path)) for scene_path in scenes_paths]
        p['scene_ids'].sort()

    scenes_paths = [os.path.join(dataset_split_path, f'{scene_id:06d}') for scene_id in p['scene_ids']]

    for scene_path in scenes_paths:  # samples are not ordered
        assembled_cloud = o3d.geometry.PointCloud()
        print("Processing", scene_path)
        scene_camera_json = os.path.join(scene_path, 'scene_camera.json')
        rgb_folder = os.path.join(scene_path, 'rgb')
        depth_folder = os.path.join(scene_path, 'depth')
        assembled_cloud_file = os.path.join(scene_path, 'assembled_cloud_world.pcd')
        with open(scene_camera_json) as j:
            scene_camera_data = json.load(j)

        # Extracting Json translation and rotation for each frame
        # for index, trans_rot in json_trot.items():
        #for index in tqdm(range(len(scene_camera_data))):
        # loop through the keys of scene_camera_data
        for index in tqdm(scene_camera_data.keys()):
            index = int(index)
            rgb_img = os.path.join(rgb_folder, f'{int(index):06}' + '.png')
            depth_img = os.path.join(depth_folder, f'{int(index):06}' + '.png')

            # Generate Homogeneous Matrix
            world2cam = np.eye(4)
            world2cam[:3, :3] = np.array(scene_camera_data[str(index)]['cam_R_w2c']).reshape(3,3)
            world2cam[:3, 3] = np.array(scene_camera_data[str(index)]['cam_t_w2c'])

            # Process one point cloud per iteration
            rgb = o3d.io.read_image(rgb_img)
            depth = o3d.io.read_image(depth_img)
            # Extract RGDB image from RGB and Depth, intensity is set to false - get colour data (3 Channels)
            depth_scale = 1/scene_camera_data[str(index)]['depth_scale']  # depth scale in mm - open3d need inverse of BOP depth scale
            # convert depth to meter
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=depth_scale,
                                                                      depth_trunc=20000,  # override default value by something big
                                                                      convert_rgb_to_intensity=False)  # rgbd in meter
            # get camera intrinsic parameters from scene_camera.json
            height, width, channels = np.asarray(rgb).shape
            fx = scene_camera_data[str(index)]['cam_K'][0]
            cx = scene_camera_data[str(index)]['cam_K'][2]
            fy = scene_camera_data[str(index)]['cam_K'][4]
            cy = scene_camera_data[str(index)]['cam_K'][5]
            camera_intrinsics= o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)

            # Open3d expects the inverse of the transform
            world2cam = np.linalg.inv(world2cam)
            transformed_cloud = pcd.transform(world2cam)
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))  # TODO recheck radius

            # assembled_cloud += transformed_cloud
            assembled_cloud += transformed_cloud.voxel_down_sample(voxel_size=p['voxel_size'])

        assembled_cloud = assembled_cloud.voxel_down_sample(voxel_size=p['voxel_size'])

        if p['show_assembled_cloud']:
            o3d.visualization.draw_geometries([assembled_cloud])

        # Save assembled cloud
        # TODO check if assembled point cloud is already there and make a warning
        o3d.io.write_point_cloud(assembled_cloud_file, assembled_cloud)


if __name__ == "__main__":
    main()
