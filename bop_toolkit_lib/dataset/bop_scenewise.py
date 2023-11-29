"""
Tools to manipulate the bop-scenewise format

bop-scenewise is the standard format described in docs/datasets_format

├─ dataset
│  ├─ SCENE_ID
│  │  ├─ scene_camera.json
│  │  ├─ scene_gt.json
│  │  ├─ scene_gt_info.json
│  │  ├─ depth
│  │  ├─ mask
│  │  ├─ mask_visib
│  │  ├─ rgb|gray
"""

import json
import pathlib
import re

import numpy as np

import bop_toolkit_lib.inout as inout


def instance_id_from_mask_filename(fn):
    """Extracts instance id from the filename of a mask
    The mask with filename '000345_000001.png' corresponds to instance id 1.

    :param fn: mask filename
    :return: instance id for this mask
    """
    return int(re.findall("\d+", fn)[-1])


def load_masks(
    scene_dir, image_id, mask_type="mask", n_instances=None, instance_ids=None
):
    """Loads one or multiple object masks stored as individual .png files.

    :param scene_dir: Path to the scene directory.
    :param image_id: Image id for which masks should be loaded.
    :param mask_type: mask or mask_visib.
    :param n_instances: Number of object instances in the scene. This can be
    used to avoid querying disks to find the number of object instances.
    This is not used when instance_ids are provided.
    :param instance_ids: List of instances ids to load the object for.
    :return: (N, H, W) numpy array with binary masks. If instance_ids are
    provided, N is the number of instances that were queried and the
    masks are in the same order as instance_ids. Otherwise, all instances
    are loaded.
    """

    if n_instances is not None and instance_ids is None:
        instance_ids = range(n_instances)

    if instance_ids is not None:
        mask_paths = (
            scene_dir / mask_type / f"{image_id:06d}_{instance_id:06d}.png"
            for instance_id in instance_ids
        )
    else:
        mask_paths = (scene_dir / mask_type).glob(f"{image_id:06d}_*.png")
        mask_paths = sorted(
            mask_paths, key=lambda p: instance_id_from_mask_filename(p.name)
        )
    masks = np.stack([inout.load_im(p) for p in mask_paths])
    return masks


def read_scene_infos(
    scene_dir,
    read_image_ids=False,
    read_n_objects=False,
):
    """Parse scene files to load information about the scene.
    Information contains binary values capturing available annotations:
        - rgb
        - depth
        - gray
        - mask
        - mask_visib
        - gt
        - gt_info,

    and optionally:
        - image_ids (list of image_ids for this scene)
        - n_objects (number of object instances in the scene).

    :param scene_dir: Path to the scene directory.
    :param read_image_ids: Read list of image ids for this scene.
    :param read_n_objects: Read number of objects in this scene.
    :return: a dict with the following keys:
        - has_rgb
        - has_depth
        - has_gray
        - has_mask
        - has_mask_visib
        - has_gt
        - has_gt_info
        - image_ids
        - n_objects.
    """

    scene_dir = pathlib.Path(scene_dir)

    infos = dict(
        has_rgb=(scene_dir / "rgb").exists(),
        has_depth=(scene_dir / "depth").exists(),
        has_gray=(scene_dir / "gray").exists(),
        has_mask=(scene_dir / "mask").exists(),
        has_mask_visib=(scene_dir / "mask_visib").exists(),
        has_gt=(scene_dir / "scene_gt.json").exists(),
        has_gt_info=(scene_dir / "scene_gt_info.json").exists(),
        image_ids=None,
        n_objects=None,
    )

    assert (scene_dir / "scene_camera.json").exists()
    if read_image_ids:
        scene_cameras = json.loads((scene_dir / "scene_camera.json").read_text())
        image_ids = [int(k) for k in scene_cameras.keys()]
        infos["image_ids"] = image_ids

    if infos["has_gt"] and read_n_objects:
        scene_gt = json.loads((scene_dir / "scene_gt.json").read_text())
        infos["n_objects"] = len(scene_gt)

    return infos


def load_scene_data(
    scene_dir,
    load_scene_camera=True,
    load_scene_gt=True,
    load_scene_gt_info=True,
):
    """Loads files with scene-level annotations.

    :param scene_dir: Path to the scene directory.
    :param load_scene_camera: load scene_camera.json.
    :param load_scene_gt: load scene_gt.json.
    :param load_scene_gt_info: load scene_gt_info.json.
    :return: a dict with keys:
        - scene_camera [Optional]
        - scene_gt [Optional]
        - scene_gt_info [Optional].
    """
    scene_data = dict(
        scene_camera=None,
        scene_gt=None,
        scene_gt_info=None,
    )

    if load_scene_camera:
        scene_camera = inout.load_scene_camera(scene_dir / "scene_camera.json")
        scene_data["scene_camera"] = scene_camera
    if load_scene_gt:
        scene_gt = inout.load_scene_gt(scene_dir / "scene_gt.json")
        scene_data["scene_gt"] = scene_gt
    if load_scene_gt_info:
        scene_gt_info = inout.load_scene_gt(scene_dir / "scene_gt_info.json")
        scene_data["scene_gt_info"] = scene_gt_info
    return scene_data


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
    """Utility to load all information about an image.
    Please note that using load_gt or load_gt_info will
    load all scene_gt.json or scene_gt_info.json repsectively, then
    select a subset corresponding to the image specified with image_id.
    Please see the v2 for more efficient individual image readings.

    :param scene_dir: Path to the scene directory.
    :param image_id: Id of an image in this scene.
    :param load_rgb: Load .png or .jpg rgb image.
    :param load_gray: Load gray .tiff image.
    :param load_depth: Load .png depth image and rescale
    it using depth_scale found in camera.json if rescale_depth=True.
    :param load_mask_visib: Load modal masks (all instances
    or only those specified by instance_ids).
    :param load_mask: Load amodal masks (all instances
    or only those specified by instance_ids).
    :param load_gt: Load ground truth object poses found
    in scene_gt.json for this image.
    :param load_gt_info: Load ground truth additional information
    found in scene_gt_info.json.
    :param rescale_depth: Whether to rescale depth to
    millimeters, defaults to True.
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

    image_data = dict(
        camera=None,
        im_rgb=None,
        im_gray=None,
        mask=None,
        mask_visib=None,
        gt=None,
        gt_info=None,
    )

    scene_dir = pathlib.Path(scene_dir)
    if isinstance(image_id, str):
        image_id = int(image_id)

    scene_cameras = inout.load_scene_camera(scene_dir / "scene_camera.json")
    camera = scene_cameras[image_id]
    n_instances = None
    image_data["camera"] = camera

    if load_rgb:
        rgb_path = scene_dir / "rgb" / f"{image_id:06d}.jpg"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".png")
        im_rgb = inout.load_im(rgb_path).astype(np.uint8)
        image_data["im_rgb"] = im_rgb

    if load_gray:
        gray_path = scene_dir / "gray" / f"{image_id:06d}.tiff"
        im_gray = inout.load_im(gray_path).astype(np.uint8)
        image_data["im_gray"] = im_gray

    if load_depth:
        depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
        im_depth = inout.load_im(depth_path).astype(np.float32)
        if rescale_depth:
            im_depth *= camera["depth_scale"]
        image_data["im_depth"] = im_depth

    if load_gt:
        scene_gt = inout.load_json(scene_dir / "scene_gt.json", keys_to_int=True)
        gt = scene_gt[image_id]
        if instance_ids is not None:
            gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
        gt = [inout._gt_as_numpy(gt_n) for gt_n in gt]
        n_instances = len(gt)
        image_data["gt"] = gt

    if load_gt_info:
        scene_gt_info = inout.load_json(
            scene_dir / "scene_gt_info.json", keys_to_int=True
        )
        gt_info = scene_gt_info[image_id]
        if instance_ids is not None:
            gt_info = [
                gt_info_n for n, gt_info_n in enumerate(gt_info) if n in instance_ids
            ]
        gt_info = [inout._gt_as_numpy(gt_info_n) for gt_info_n in gt_info]
        n_instances = len(gt_info)
        image_data["gt_info"] = gt_info

    if load_mask_visib:
        mask_visib = load_masks(
            scene_dir,
            image_id,
            mask_type="mask_visib",
            n_instances=n_instances,
            instance_ids=instance_ids,
        )
        image_data["mask_visib"] = mask_visib

    if load_mask:
        mask = load_masks(
            scene_dir,
            image_id,
            mask_type="mask",
            n_instances=n_instances,
            instance_ids=instance_ids,
        )
        image_data["mask"] = mask

    return image_data
