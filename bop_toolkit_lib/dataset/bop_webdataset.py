import json
import tarfile

import numpy as np

from bop_toolkit_lib import inout
from bop_toolkit_lib.dataset import bop_v2


def load_image_data(
    shard_path,
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
    rgb_suffix='.jpg',
):

    tar = tarfile.open(shard_path, mode='r')

    def _load(ext, read=True):
        buffered_reader = tar.extractfile(f'{image_key}.{ext}')
        if read:
            return buffered_reader.read()
        else:
            return buffered_reader

    image_data = dict(
        camera=None,
        im_rgb=None,
        im_gray=None,
        mask=None,
        mask_visib=None,
        gt=None,
        gt_info=None,
    )
    camera = json.load(_load('camera.json', read=False))
    image_data['camera'] = camera

    if load_rgb:
        image_data['im_rgb'] = inout.load_im(
            _load('rgb' + rgb_suffix)).astype(np.uint8)

    if load_gray:
        image_data['im_gray'] = inout.load_im(
            _load('gray.tiff')).astype(np.uint8)

    if load_depth:
        im_depth = inout.load_im(
            _load('depth.png')).astype(np.float32)
        if rescale_depth:
            im_depth *= camera['depth_scale']
        image_data['im_depth'] = im_depth

    if load_gt:
        image_data['gt'] = bop_v2.io_load_gt(
            _load('gt.json', read=False),
            instance_ids=instance_ids)

    if load_gt_info:
        image_data['gt_info'] = bop_v2.io_load_gt(
            _load('gt_info.json', read=False),
            instance_ids=instance_ids)

    if load_mask_visib:
        image_data['mask_visib'] = bop_v2.io_load_masks(
            _load('mask_visib.json', read=False),
            instance_ids=instance_ids)

    if load_mask:
        image_data['mask'] = bop_v2.io_load_masks(
            _load('mask.json', read=False),
            instance_ids=instance_ids)

    tar.close()
    return image_data
