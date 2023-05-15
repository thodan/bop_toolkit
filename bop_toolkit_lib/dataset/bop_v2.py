import pathlib
import numpy as np
import json
from bop_toolkit_lib import inout
from bop_toolkit_lib import pycoco_utils


def _save_scene_dict(
    scene_dict,
    image_tpath,
    json_converter,
):
    for image_id, image_dict in scene_dict.items():
        image_dict = json_converter(image_dict)
        path = image_tpath.format(image_id=image_id)
        inout.save_json(path, image_dict)
    return


def save_scene_camera(
    scene_camera,
    image_camera_tpath,
):
    _save_scene_dict(
        scene_camera,
        image_camera_tpath,
        inout._camera_as_json
    )
    return


def save_scene_gt(
    scene_gt,
    image_gt_tpath,
):
    _save_scene_dict(
        scene_gt,
        image_gt_tpath,
        lambda lst: [inout._gt_as_json(d) for d in lst]
    )
    return


def save_masks(
    masks,
    masks_path,
):
    masks_rle = dict()
    for instance_id, mask in enumerate(masks):
        mask_rle = pycoco_utils.binary_mask_to_rle(mask)
        masks_rle[instance_id] = mask_rle
    inout.save_json(masks_path, masks_rle)
    return


def io_load_masks(
    mask_file,
    instance_ids=None
):
    masks_rle = json.load(mask_file)
    if instance_ids is not None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack([
        pycoco_utils.rle_to_binary_mask(mask_rle)[:, :, None]
        for mask_rle in masks_rle], axis=-1)
    return masks


def io_load_gt(
    gt_file,
    instance_ids=None,
):
    gt = json.load(gt_file)
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [inout._gt_as_json(gt_n) for gt_n in gt]
    return gt


def load_image_infos(
    dataset_dir,
    image_key,
):
    def _file_path(ext):
        return dataset_dir / f'{image_key}.{ext}'

    infos = dict()
    dataset_dir = pathlib.Path(dataset_dir)
    has_rgb = _file_path('rgb.png').exists()
    has_rgb = has_rgb or _file_path('rgb.jpg').exists()
    infos['has_rgb'] = has_rgb
    infos['has_depth'] = _file_path('depth.png').exsits()
    infos['has_gray'] = _file_path('gray.tiff').exists()
    infos['has_mask'] = _file_path('mask.png').exists()
    infos['has_mask_visib'] = _file_path('mask_visib.png').exists()
    infos['has_gt'] = _file_path('gt.json').exists()
    infos['has_gt_info'] = _file_path('gt_info.json').exists()
    return infos


def load_image_data(
    dataset_dir,
    image_key,
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

    def _file_path(ext):
        return dataset_dir / f'{image_key}.{ext}'

    image_data = dict()

    camera = inout.load_json(_file_path('camera.json'))
    camera = inout._camera_as_numpy(camera)
    image_data['camera'] = camera

    if load_rgb:
        rgb_path = _file_path('rgb.jpg')
        if not rgb_path.exists():
            rgb_path = _file_path('rgb.png')
        image_data['im_rgb'] = inout.load_im(rgb_path).astype(np.uint8)

    if load_gray:
        gray_path = _file_path('gray.tiff') 
        im_gray = inout.load_im(gray_path).astype(np.uint8)
        image_data['im_gray'] = im_gray
    
    if load_depth:
        depth_path = _file_path('depth.png')
        im_depth = inout.load_im(depth_path).astype(np.float32)
        if rescale_depth:
            im_depth *= camera['depth_scale']
        image_data['im_depth'] = im_depth

    if load_gt:
        with open(_file_path('gt.json'), 'r') as f:
            image_data['gt'] = io_load_gt(f, instance_ids=instance_ids)

    if load_gt_info:
        with open(_file_path('gt_info.json'), 'r') as f:
            image_data['gt_info'] = io_load_gt(f, instance_ids=instance_ids)

    if load_mask_visib:
        with open(_file_path('mask_visib.json'), 'r') as f:
            image_data['mask_visib'] = io_load_masks(
                f, instance_ids=instance_ids)

    if load_mask:
        with open(_file_path('mask.json'), 'r') as f:
            image_data['mask'] = io_load_masks(
                f, instance_ids=instance_ids)

    return image_data