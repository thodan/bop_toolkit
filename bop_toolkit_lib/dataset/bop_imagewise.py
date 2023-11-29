"""
Tools to manipulate the bop-imagewise format

bop-imagewise is a format where the image annotations are stored
in individual files. This format is only used as an intermediate step
to convert a bop-scenewise dataset to a bop-webdataset.
Format is the following:

├─ dataset
│  ├─ KEY.{rgb|gray}.{png|jpg}
│  ├─ KEY.depth.png
│  ├─ KEY.camera.json
│  ├─ KEY.gt.json
│  ├─ KEY.gt_info.json
│  ├─ KEY.mask.json
│  ├─ KEY.mask_visib.json
    ... ,
where KEY is a unique identifier of an image in the dataset.
Typically it is {scene_id:06d}_{image_id:06d}.
"""

import json
import pathlib

import numpy as np

from bop_toolkit_lib import inout, pycoco_utils


def _save_scene_dict(
    scene_dict,
    image_tpath,
    json_converter,
):
    """Saves a scene dict annotation as
    individual files (one for each image).

    :param scene_dict: dict that has keys
    corresponding to image ids and values corresponding
    to a dictionary of image annotations.
    :param image_tpath: template path with unspecified image_id.
    :param json_converter: a function that converts the
    image annotations to json.
    """
    for image_id, image_dict in scene_dict.items():
        image_dict = json_converter(image_dict)
        path = str(image_tpath).format(image_id=image_id)
        inout.save_json(path, image_dict)
    return


def save_scene_camera(
    scene_camera,
    image_camera_tpath,
):
    """Saves scene_camera
    (typically found in scene_camera.json
    in the BOP-scenewise format) to individual files.

    :param scene_camera: scene_camera
    dict mapping image_ids to camera information.
    :param image_camera_tpath: template path with unspecified image_id.
    """
    _save_scene_dict(scene_camera, image_camera_tpath, inout._camera_as_json)
    return


def save_scene_gt(
    scene_gt,
    image_gt_tpath,
):
    """Saves scene ground truth
    (typically found in scene_gt.json or
    scene_gt_info.json in the BOP-scenewise format) to individual files.

    :param scene_camera: scene_gt
    dict mapping image_ids to gt information.
    :param image_camera_tpath: template path with unspecified image_id.
    """
    _save_scene_dict(
        scene_gt, image_gt_tpath, lambda lst: [inout._gt_as_json(d) for d in lst]
    )
    return


def save_masks(
    masks,
    masks_path,
):
    """Saves object masks to a single file.
    The object masks are RLE encoded and written in json.
    The json file contains a dict mapping instance ids
    to RLE data ('counts' and 'size').

    :param masks: [N,H,W] binary numpy arrays,
    where N is the number of object instances.
    :param masks_path: Path to json file.
    """
    masks_rle = dict()
    for instance_id, mask in enumerate(masks):
        mask_rle = pycoco_utils.binary_mask_to_rle(mask)
        masks_rle[instance_id] = mask_rle
    inout.save_json(masks_path, masks_rle)
    return


def io_load_masks(mask_file, instance_ids=None):
    """Load object masks from an I/O object.
    Instance_ids can be specified to apply RLE
    decoding to a subset of object instances contained
    in the file.

    :param mask_file: I/O object that can be read with json.load.
    :param masks_path: Path to json file.
    :return: a [N,H,W] binary array containing object masks.
    """
    masks_rle = json.load(mask_file)
    masks_rle = {int(k): v for k, v in masks_rle.items()}
    if instance_ids is None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack(
        [
            pycoco_utils.rle_to_binary_mask(masks_rle[instance_id])
            for instance_id in instance_ids
        ]
    )
    return masks


def io_load_gt(
    gt_file,
    instance_ids=None,
):
    """Load ground truth from an I/O object.
    Instance_ids can be specified to load only a
    subset of object instances.

    :param gt_file: I/O object that can be read with json.load.
    :param instance_ids: List of instance ids.
    :return: List of ground truth annotations (one dict per object instance).
    """
    gt = json.load(gt_file)
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [inout._gt_as_numpy(gt_n) for gt_n in gt]
    return gt


def load_image_infos(
    dataset_dir,
    image_key,
):
    """Parse files to read information about the image.

    :param dataset_dir: path to a dataset directory.
    :param image_key: string that uniqly identifies the image in the dataset.
    """

    def _file_path(ext):
        return dataset_dir / f"{image_key}.{ext}"

    infos = dict(
        has_rgb=False,
        has_depth=_file_path("depth.png").exists(),
        has_gray=_file_path("gray.tiff").exists(),
        has_mask=_file_path("mask.json").exists(),
        has_mask_visib=_file_path("mask_visib.json").exists(),
        has_gt=_file_path("gt.json").exists(),
        has_gt_info=_file_path("gt_info.json").exists(),
    )

    if _file_path("rgb.png").exists():
        infos["has_rgb"] = True
        infos["rgb_suffix"] = ".png"

    if _file_path("rgb.jpg").exists():
        assert "rgb_suffix" not in infos
        infos["has_rgb"] = True
        infos["rgb_suffix"] = ".jpg"

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
    """Utility to load all information about an image.

    :param dataset_dir: Path to a dataset directory.
    :param image_key: string that uniqly identifies the image in the dataset.
    :param load_rgb: load {image_key}.rgb.png or {image_key}.rgb.jpg.
    :param load_gray: load {image_key}.gray.tiff
    :param load_depth: load {image_key}.depth.png and rescale
    it using depth_scale found in {image_key}.camera.json
    if rescale_depth=True.
    :param load_mask_visib: Load modal masks found in
    {image_key}.mask_visib.png (all instances
    or only those specified by instance_ids).
    :param load_mask: Load amodal masks found in
    {image_key}.mask.png (all instances
    or only those specified by instance_ids).
    :param load_gt: load ground truth object poses found in
    {image_key}.gt.json.
    :param load_gt_info: Load ground truth additional information
    found in {image_key}.gt_info.json.
    :param rescale_depth:  Whether to rescale the depth
    image to millimeters, defaults to True.
    :param instance_ids: List of instance ids,
    used to restrict loading to a subset of object masks.
    :return: A dictionary with the following content:
        - camera
        - im_rgb
        - im_gray
        - im_depth
        - mask
        - mask_visib
        - gt
        - gt_info.
    """

    dataset_dir = pathlib.Path(dataset_dir)

    def _file_path(ext):
        return dataset_dir / f"{image_key}.{ext}"

    image_data = dict(
        camera=None,
        im_rgb=None,
        im_gray=None,
        mask=None,
        mask_visib=None,
        gt=None,
        gt_info=None,
    )

    camera = inout.load_json(_file_path("camera.json"))
    camera = inout._camera_as_numpy(camera)
    image_data["camera"] = camera

    if load_rgb:
        rgb_path = _file_path("rgb.jpg")
        if not rgb_path.exists():
            rgb_path = _file_path("rgb.png")
        image_data["im_rgb"] = inout.load_im(rgb_path).astype(np.uint8)

    if load_gray:
        gray_path = _file_path("gray.tiff")
        im_gray = inout.load_im(gray_path).astype(np.uint8)
        image_data["im_gray"] = im_gray

    if load_depth:
        depth_path = _file_path("depth.png")
        im_depth = inout.load_im(depth_path).astype(np.float32)
        if rescale_depth:
            im_depth *= camera["depth_scale"]
        image_data["im_depth"] = im_depth

    if load_gt:
        with open(_file_path("gt.json"), "r") as f:
            image_data["gt"] = io_load_gt(f, instance_ids=instance_ids)

    if load_gt_info:
        with open(_file_path("gt_info.json"), "r") as f:
            image_data["gt_info"] = io_load_gt(f, instance_ids=instance_ids)

    if load_mask_visib:
        with open(_file_path("mask_visib.json"), "r") as f:
            image_data["mask_visib"] = io_load_masks(f, instance_ids=instance_ids)

    if load_mask:
        with open(_file_path("mask.json"), "r") as f:
            image_data["mask"] = io_load_masks(f, instance_ids=instance_ids)

    return image_data
