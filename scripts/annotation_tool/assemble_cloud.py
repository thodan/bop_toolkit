"""
This script will assemble a point cloud of consisting of all frames in each sample of the dataset.
This is required for annotating datasets that are captured in a sequence of frames.
All frames in a sample are assembled into on big point cloud in the first frame.

To be able to run this script scene_camera.json must include "cam_R_w2c" and "cam_t_w2c".
The script will save the assembled point cloud as assembled_cloud_WORLD.pcd in the world_frame in each sample folder.

All units in this script are mm as in the BOP standard. The assembled point cloud is also saved in mm.
"""

import copy
import open3d as o3d
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import os
import glob

# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/run/user/1000/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/DoPose_v2/dataset',

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': 'train',

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': None,

    # Refine the transformations between point clouds. Options: True, False
    # WARNING: when this option is used the original scene_camera.json will be renamed scene_camera_UNREFINED.json
    #          and the new refined transformation will be added as scene_camera.json
    # This option require CUDA. unfortunately the ICP is very slow without CUDA and it is not recomended.
    'refine': False,
    'show_refined_cloud': True
}
################################################################################


def main():
    dataset_split_path = os.path.join(
        p['dataset_path'],
        p['dataset_split'] + '_' + p['dataset_split_type'] if p['dataset_split_type'] else p['dataset_split'])
    scenes_paths = glob.glob(dataset_split_path + '/*')

    assembled_cloud = o3d.geometry.PointCloud()
    for scene_path in scenes_paths:  # samples are not ordered
        scene_camera_json = os.path.join(scene_path, 'scene_camera.json')
        rgb_folder = os.path.join(scene_path, 'rgb')
        depth_folder = os.path.join(scene_path, 'depth')
        assembled_cloud_file = os.path.join(scene_path, 'assembled_cloud.pcd')
        with open(scene_camera_json) as j:
            scene_camera_data = json.load(j)

        clouds = []
        # Extracting Json translation and rotation for each frame
        # for index, trans_rot in json_trot.items():
        for index in range(len(scene_camera_data)):
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
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1000,
                                                                      convert_rgb_to_intensity=False)  # rgbd in meters
            # Intrinsic parameters is constant for every frame and scene
            camera_intrinsic_zivid = o3d.camera.PinholeCameraIntrinsic(width=1944, height=1200,
                                                                       fx=1778.81005859375, fy=1778.87036132812,
                                                                       cx=967.931579589844, cy=572.408813476562)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic_zivid)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000)  # convert point cloud to mm

            transformed_cloud = pcd.transform(world2cam)
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            #o3d.visualization.draw_geometries([transformed_pcd])
            clouds.append(transformed_cloud)
            assembled_cloud += transformed_cloud
        #o3d.visualization.draw_geometries(clouds)

        if p['refine']:  # NOTE: ICP in Open3d have to be done with all units in meter. Convert all to meter and convert back to mm.
            # Convert point clouds from legacy to tensors
            clouds_t = []
            for cloud in clouds:  # TODO remove this loop do everything in the outer loop
                cloud_copy = copy.deepcopy(cloud)
                cloud_copy.points = o3d.utility.Vector3dVector(np.asarray(cloud_copy.points) / 1000)  # convert cloud in meter for icp
                cloud_t = o3d.t.geometry.PointCloud.from_legacy(cloud_copy)
                cloud_t.point["colors"] = cloud_t.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
                cloud_t = cloud_t.cuda(0)
                clouds_t.append(cloud_t)

            # align all clouds to first cloud using ICP - update the transformation of VICON
            import open3d.t.pipelines.registration as treg
            max_correspondence_distances = 0.02  # in meter
            # TODO test multi-scale voxel size
            voxel_size = 0.001  # in meter
            criteria = treg.ICPConvergenceCriteria(relative_fitness=1.000000e-06,relative_rmse=1.000000e-06,max_iteration=300)

            refined_clouds = [clouds[0]]
            target_cloud = clouds_t[0]
            assembled_cloud = clouds[0]
            for idx, source_cloud in enumerate(clouds_t[1:]):
                cloud_idx = idx + 1  # as we don't iterate from the first sample
                print("assembling cloud ", str(cloud_idx), " with source (first) cloud")
                reg_transform = treg.icp(source_cloud, target_cloud,
                                         max_correspondence_distances,
                                         estimation_method=treg.TransformationEstimationForColoredICP(),
                                         criteria=criteria, voxel_size=voxel_size)

                icp_transform = reg_transform.transformation.cpu().numpy() # Convert tensor to numpy array
                # Generate Homogeneous Matrix
                world2cam = np.eye(4)
                world2cam[:3, :3] = np.array(scene_camera_data[str(index)]['cam_R_w2c']).reshape(3, 3)
                world2cam[:3, 3] = np.array(scene_camera_data[str(index)]['cam_t_w2c'])
                refined_transform = np.matmul(world2cam, icp_transform)

                icp_transform[:3, 3] *= 1000  # convert the transformation from meter to mm
                transformed_source = clouds[cloud_idx].transform(icp_transform)
                refined_clouds.append(transformed_source)
                assembled_cloud += transformed_source

                scene_camera_data[str(cloud_idx)]['cam_t_w2c'] = refined_transform[:3, 3].tolist()  # TODO should that be flattened?
                scene_camera_data[str(cloud_idx)]['cam_R_w2c'] = refined_transform[:3,:3].flatten().tolist()

            if p['show_refined_cloud']:
                o3d.visualization.draw_geometries(refined_clouds)

        # Save assembled cloud
        # TODO remove redundant points in the assembled clouds
        # TODO check if assembled point cloud is already there and make a warning
        o3d.io.write_point_cloud(assembled_cloud_file, assembled_cloud)
        if p['refine']:
            os.rename(scene_camera_json, scene_camera_json[:-5]+'_UNREFINED.json')
            with open(scene_camera_json, 'w') as f:
                json.dump(scene_camera_data, f)

if __name__ == "__main__":
    main()
