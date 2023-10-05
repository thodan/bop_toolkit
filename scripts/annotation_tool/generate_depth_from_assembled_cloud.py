import json
import open3d as o3d
import argparse
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm

from open3d.visualization import Visualizer

"""
This script will generate depth images from the assembled point cloud.
These generated depth images will help generating better annotations compared to original depth images.
This is mainly becuase these generated depth images have less holes compared to the original depth images,
due to the multi-view assembly of the assembled point cloud.
"""

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
    if p['dataset_split_type'] is not None:
        p['dataset_split'] = p['dataset_split'] + '_' + p['dataset_split_type']
    samples = glob.glob(p['dataset_path'] + '/' + p['dataset_split'] + '/*')

    # create window
    vis = Visualizer()
    # image size from Zivid camera
    w = 1944
    h = 1200
    vis.create_window(width=w, height=h, visible=False)

    # set point size in the visualizer to 1
    opt = vis.get_render_option()
    opt.point_size = 3
    opt.show_coordinate_frame = True

    for sample in samples:
        print('Processing sample: ', sample)

        cloud_file = sample + '/assembled_cloud_WORLD.pcd'
        cloud = o3d.io.read_point_cloud(cloud_file)

        # change point cloud from mm to meter
        #cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points) / 1000)
        # if this is uncommented the extrinsics also needs to be changed to meter

        # make generated depth folder
        depth_folder = sample + '/depth_generated'
        if not os.path.exists(depth_folder):
            os.mkdir(depth_folder)

        # load scene_camera.json
        scene_camera_json = sample + '/scene_camera.json'
        with open(scene_camera_json) as j:
            scene_camera_data = json.load(j)

        vis.add_geometry(cloud, reset_bounding_box=True)

        #vis.run()

        for frame_idx, cam_info in tqdm(scene_camera_data.items()):
            # get camera intrinsics
            cam_K = np.array(cam_info['cam_K']).reshape(3,3)
            # get camera extrinsics
            cam_extrinsic = np.eye(4)
            cam_extrinsic[:3,:3] = np.array(cam_info['cam_R_w2c']).reshape(3,3).transpose()
            cam_extrinsic[:3,3] = - cam_extrinsic[:3,:3] @ np.array(cam_info['cam_t_w2c'])

            # change camera pose to loaded value from json
            ctr = vis.get_view_control()
            # make pinhole camera parameters
            pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2])
            pinhole_parameters = o3d.camera.PinholeCameraParameters()
            #pinhole_parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
            pinhole_parameters.extrinsic = cam_extrinsic
            pinhole_parameters.intrinsic = pinhole_intrinsics
            ctr.convert_from_pinhole_camera_parameters(pinhole_parameters, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()

            #vis.run()
            #vis.destroy_window()

            # capture depth image
            depth = vis.capture_depth_float_buffer(do_render=False)

            # save depth image
            depth_file = depth_folder + '/' + f'{int(frame_idx):06}' + '.png'
            #o3d.io.write_image(depth_file, depth)
            depth_img_cv2 = np.array(depth, dtype=np.uint16)
            cv2.imwrite(depth_file, depth_img_cv2)

        vis.remove_geometry(cloud)

if __name__ == '__main__':
    main()