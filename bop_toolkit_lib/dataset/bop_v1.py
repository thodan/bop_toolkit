import re
import numpy as np
import json
import pathlib
import bop_toolkit_lib.inout as inout


def instance_id_from_mask_filename(fn):
    return int(re.findall("\d+", fn)[-1])


def load_masks(
    scene_dir,
    image_id,
    mask_type='mask',
    n_instances=None,
    instance_ids=None
):

    if n_instances is not None and instance_ids is None:
        instance_ids = range(n_instances)

    if instance_ids is not None:
        mask_paths = (
            scene_dir / mask_type / f'{image_id:06d}_{instance_id}.png'
            for instance_id in instance_ids)
    else:
        mask_paths = (
            scene_dir / mask_type).glob(f'{image_id:06d}_*.png')
        mask_paths = sorted(
            mask_paths,
            key=lambda p: instance_id_from_mask_filename(p.name)
        )
    masks = np.stack([
        inout.load_im(p) for p in mask_paths
    ], axis=-1)
    return masks


def read_scene_infos(
    scene_dir,
    read_image_ids=True,
    read_n_objects=True,
):
    # Outputs number of scenes, image ids for each scene.
    scene_dir = pathlib.Path(scene_dir)

    infos = dict()
    infos['has_rgb'] = (scene_dir / 'rgb').exists()
    infos['has_depth'] = (scene_dir / 'depth').exists()
    infos['has_gray'] = (scene_dir / 'gray').exists()
    infos['has_mask'] = (scene_dir / 'masks').exists()
    infos['has_mask_visib'] = (scene_dir / 'mask_visib').exists()

    infos['has_gt'] = (scene_dir / 'scene_gt.json').exists()
    infos['has_gt_info'] = (scene_dir / 'scene_gt_info.json').exists()
    assert (scene_dir / 'scene_camera.json').exists()

    if read_image_ids:
        scene_cameras = json.loads((scene_dir / 'scene_camera.json').read_text())
        image_ids = [int(k) for k in scene_cameras.keys()]
        infos['image_ids'] = image_ids

    if infos['has_gt'] and read_n_objects:
        scene_gt = json.loads((scene_dir / 'scene_gt.json').read_text())
        infos['n_objects'] = len(scene_gt)

    return infos


def load_scene_data(
    scene_dir,
    load_scene_camera=True,
    load_scene_gt=True,
    load_scene_gt_info=True,
):
    scene_data = dict()
    if load_scene_camera:
        scene_camera = inout.load_scene_camera(scene_dir / 'scene_camera.json')
        scene_data['scene_camera'] = scene_camera
    if load_scene_gt:
        scene_gt = inout.load_scene_gt(scene_dir / 'scene_gt.json')
        scene_data['scene_gt'] = scene_gt
    if load_scene_gt_info:
        scene_gt_info = inout.load_scene_gt(scene_dir / 'scene_gt_info.json')
        scene_data['scene_gt_info'] = scene_gt_info
    return


def load_image_data(
    scene_dir,
    image_id,
    load_rgb=True,
    load_gray=False,
    load_depth=True,
    load_mask_visib=True,
    load_mask=False,
    load_gt=False,
    load_gt_info=False,
    rescale_depth=True,
    instance_ids=None,
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

    image_data = dict()

    scene_dir = pathlib.Path(scene_dir)
    if isinstance(image_id, str):
        image_id = int(image_id)

    scene_cameras = inout.load_scene_camera(scene_dir / 'scene_camera.json')
    camera = scene_cameras[image_id]
    image_data['camera'] = camera

    if load_rgb:
        rgb_path = scene_dir / 'rgb' / f'{image_id:06d}.jpg'
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix('.png')
        im_rgb = inout.load_im(rgb_path).astype(np.uint8)
        image_data['im_rgb'] = im_rgb

    if load_gray:
        gray_path = scene_dir / 'gray' / f'{image_id:06d}.tiff'
        im_gray = inout.load_im(gray_path).astype(np.uint8)
        image_data['im_gray'] = im_gray
    
    if load_depth:
        depth_path = scene_dir / 'depth' / f'{image_id:06d}.png'
        im_depth = inout.load_im(depth_path).astype(np.float32)
        if rescale_depth:
            im_depth *= camera['depth_scale']
        image_data['im_depth'] = im_depth

    if load_gt:
        scene_gt = inout.load_json(scene_dir / 'scene_gt.json')
        gt = scene_gt[image_id]
        if instance_ids is not None:
            gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
        gt = [inout._gt_as_json(gt_n) for gt_n in gt]
        image_data['gt'] = gt

    if load_gt_info:
        scene_gt_info = inout.load_json(
            scene_dir / 'scene_gt_info.json', keys_to_int=True)
        gt_info = scene_gt_info[image_id]
        if instance_ids is not None:
            gt_info = [
                gt_info_n for n, gt_info_n in enumerate(gt_info)
                if n in instance_ids]
        gt_info = [inout._gt_as_json(gt_info) for gt_info_n in gt_info]
        image_data['gt_info'] = inout._gt_as_json(gt_info)

    if load_mask_visib:
        mask_visib = load_masks(
            scene_dir,
            image_id,
            mask_type='mask_visib',
            n_instances=len(gt) if gt is not None else None,
            instance_ids=instance_ids
        )
        image_data['mask_visib'] = mask_visib

    if load_mask:
        mask = load_masks(
            scene_dir,
            image_id,
            mask_type='mask',
            n_instances=len(gt) if gt is not None else None,
            instance_ids=instance_ids
        )
        image_data['mask'] = mask

    return image_data
