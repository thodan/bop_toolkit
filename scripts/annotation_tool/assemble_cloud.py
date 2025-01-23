"""
This script will assemble a point cloud of consisting of all frames in each sample of the dataset.
This is required for annotating datasets that are captured in a sequence of frames.
All frames in a sample are assembled into on big point cloud in the first frame.

To be able to run this script scene_camera.json must include "cam_R_w2c" and "cam_t_w2c".
The script will save the assembled point cloud as assembled_cloud_WORLD.pcd in the world_frame in each sample folder.

All units in this script are mm as in the BOP standard. The assembled point cloud is also saved in mm.
"""

import open3d as o3d
import copy
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import os
import glob
from tqdm import tqdm

import open3d.t.pipelines.registration as treg

# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/path/to/dataset',

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': 'train',

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': None,

    # Refine the transformations between point clouds. Options: True, False
    # WARNING: when this option is used the original scene_camera.json will be renamed scene_camera_UNREFINED.json
    #          and the new refined transformation will be added as scene_camera.json
    # This option require CUDA. unfortunately the ICP is very slow without CUDA and cannot be practically used for this task.
    'refine': False,

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
    scenes_paths = glob.glob(dataset_split_path + '/*')

    if p['refine']:  # set parameters for ICP if used
        # NOTE: ICP parameters and point clouds are in mm
        estimation = treg.TransformationEstimationForColoredICP()
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.00001,
                                               relative_rmse=0.00001,
                                               max_iteration=300)
        max_correspondence_distance = 1.5  # mm
        voxel_size = 0.001 * 1000  # TODO test with 0.5 mm voxel size
        init_source_to_target = np.eye(4)
        callback_after_iteration = lambda updated_result_dict: print(
            "Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                updated_result_dict["iteration_index"].item(),
                updated_result_dict["fitness"].item(),
                updated_result_dict["inlier_rmse"].item()))
        accumulated_target_icp_tf = np.eye(4)

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
            depth_scale = scene_camera_data[str(index)]['depth_scale'] / 1000  # depth scale converts image to meter
            depth_scale = 1/depth_scale  # depth scale for open3d is the inverse of the depth scale in BOP format
            # convert depth to meter
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=depth_scale,
                                                                      convert_rgb_to_intensity=False)  # rgbd in meter
            # get camera intrinsic parameters from scene_camera.json
            height, width, channels = np.asarray(rgb).shape
            fx = scene_camera_data[str(index)]['cam_K'][0]
            cx = scene_camera_data[str(index)]['cam_K'][2]
            fy = scene_camera_data[str(index)]['cam_K'][4]
            cy = scene_camera_data[str(index)]['cam_K'][5]
            camera_intrinsics= o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000)  # convert point cloud to mm

            # Open3d expects the inverse of the transform
            world2cam = np.linalg.inv(world2cam)
            transformed_cloud = pcd.transform(world2cam)
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))  # TODO recheck radius

            if p['refine'] and index == 0:
                previous_cloud = copy.deepcopy(transformed_cloud)  # cache current cloud as previous_cloud
            elif p['refine']:  # refine the transform between the current cloud and the previous cloud, first cloud is not refined of course
                # visualize both clouds before ICP
                #o3d.visualization.draw_geometries([transformed_cloud, previous_cloud])
                source_idx = index
                target_idx = index - 1

                target_cloud_t = cloud_gpu(previous_cloud)
                source_cloud_t = cloud_gpu(transformed_cloud)
                print("refining cloud ", str(source_idx), "(source cloud) to cloud ", str(target_idx), " (target cloud).")
                reg_transform = treg.icp(source_cloud_t,target_cloud_t,
                                         max_correspondence_distance,init_source_to_target,
                                         estimation,criteria,
                                         voxel_size)
                icp_transform = reg_transform.transformation.cpu().numpy() # Convert tensor to numpy array

                # visualize the result of ICP
                #tmp_cloud = copy.deepcopy(transformed_cloud)
                #o3d.visualization.draw_geometries([tmp_cloud.transform(icp_transform), previous_cloud])

                previous_cloud = copy.copy(transformed_cloud)  # cache current cloud as previous_cloud before it is refined

                # concatenate refined source cloud into assembled_cloud
                accumulated_target_icp_tf = icp_transform @ accumulated_target_icp_tf
                transformed_cloud = transformed_cloud.transform(accumulated_target_icp_tf)

                # Homogeneous matrix of camera from world
                source2cam = np.eye(4)
                source2cam[:3, :3] = np.array(scene_camera_data[str(source_idx)]['cam_R_w2c']).reshape(3, 3)
                source2cam[:3, 3] = np.array(scene_camera_data[str(source_idx)]['cam_t_w2c'])
                # refined transform is multiplication of the transform of the source (first) frame and the ICP refinement
                refined_transform = accumulated_target_icp_tf @ source2cam
                scene_camera_data[str(source_idx)]['cam_t_w2c'] = refined_transform[:3, 3].tolist()
                scene_camera_data[str(source_idx)]['cam_R_w2c'] = refined_transform[:3,:3].flatten().tolist()


            assembled_cloud += transformed_cloud

        assembled_cloud = assembled_cloud.voxel_down_sample(voxel_size=1)

        if p['show_assembled_cloud']:
            o3d.visualization.draw_geometries([assembled_cloud])

        # Save assembled cloud
        # TODO check if assembled point cloud is already there and make a warning
        o3d.io.write_point_cloud(assembled_cloud_file, assembled_cloud)
        if p['refine']:
            os.rename(scene_camera_json, scene_camera_json[:-5]+'_UNREFINED.json')
            with open(scene_camera_json, 'w') as f:
                json.dump(scene_camera_data, f)

if __name__ == "__main__":
    main()
