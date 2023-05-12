from pathlib import Path
import numpy as np
import json
from bop_toolkit_lib.inout import (
    load_im,
    load_depth,
    load_cam_params,
    load_scene_camera,
)


def make_scene_infos(
    scene_dir,
    read_image_ids=True,
    read_n_instances=True,
):
    # Outputs number of scenes, image ids for each scene.
    scene_dir = Path(scene_dir)

    infos = dict()
    infos['has_rgb'] = (scene_dir / 'rgb').exists()
    infos['has_depth'] = (scene_dir / 'depth').exists()
    infos['has_gray'] = (scene_dir / 'gray').exists()
    infos['has_mask'] = (scene_dir / 'masks').exists()
    infos['has_mask_visib'] = (scene_dir / 'mask_visib').exists()

    infos['has_gt'] = (scene_dir / 'scene_gt.json').exists()
    infos['has_gt_info'] = (scene_dir / 'scene_gt_info.json').exists()
    assert scene_dir / 'scene_camera.json'.exists()

    return infos

def get_scene_info(
    scene_dir,
):
    return {
        'has_depth': ,
        'has_rgb': ,
        'has_gray': ,
            
    }


def load_image_data(
    scene_dir,
    image_id,
    load_rgb,
    load_gray,
    load_depth,
    load_mask_visib,
    load_mask,
    load_scene_gt=
    scale_depth=True,
    instance_id=None,
):
    """Loads all data for one image including images and annotations


    :param scene_dir: _description_
    :param image_id: _description_
    :param instance_ids: _description_, defaults to None
    :param load_masks_visib: _description_, defaults to True
    :param load_masks: _description_, defaults to True
    :param load_depth: _description_, defaults to True
    :param load_rgb: _description_, defaults to True
    :return: _description_
    """

    output = dict()

    scene_dir = Path(scene_dir)
    if isinstance(image_id, str):
        image_id = int(image_id)
        
    scene_cameras = load_scene_camera(scene_dir / 'scene_camera.json')
    camera = scene_cameras[str(image_id)]
    output['camera'] = camera

    if load_rgb:
        rgb_path = scene_dir / 'rgb' / f'{image_id:06d}.jpg'
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix('.png')
        im_rgb = load_im(rgb_path).astype(np.uint8)
        output['im_rgb'] = im_rgb

    if load_gray:
        gray_path = scene_dir / 'gray' / f'{image_id:06d}.tiff'
        im_gray = load_im(gray_path).astype(np.uint8)
        output['im_gray'] = im_gray
    
    if load_depth:
        depth_path = scene_dir / 'depth' / f'{image_id:06d}.png'
        im_depth = load_im(depth_path).astype(np.float32)
        if scale_depth:
            im_depth *= camera['depth_scale']
        output['im_depth'] = im_depth
    
    if load_masks_visib:
        if instance_ids is None:
            
    return x

    
def load_mesh_bop(
    models_dir,
):
    return


def load_mesh_shapenet(
    models_dir,
):
    return


def load_mesh_gso(
    models_dir,
):
    return