# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Parameters of the BOP datasets."""

import math
import glob
import os
from os.path import join

from bop_toolkit_lib import inout


def get_camera_params(datasets_path, dataset_name, cam_type=None):
    """Returns camera parameters for the specified dataset.

    Note that parameters returned by this functions are meant only for simulation
    of the used sensor when rendering training images. To get per-image camera
    parameters (which may vary), use path template 'scene_camera_tpath' contained
    in the dictionary returned by function get_split_params.

    :param datasets_path: Path to a folder with datasets.
    :param dataset_name: Name of the dataset for which to return the parameters.
    :param cam_type: Type of camera.
    :return: Dictionary with camera parameters for the specified dataset.
    """
    if dataset_name == "tless":
        # Includes images captured by three sensors. Use Primesense as default.
        if cam_type is None:
            cam_type = "primesense"
        cam_filename = "camera_{}.json".format(cam_type)

    elif dataset_name in ["hbs", "hb"]:
        # Both versions of the HB dataset share the same directory.
        dataset_name = "hb"

        # Includes images captured by two sensors. Use Primesense as default.
        if cam_type is None:
            cam_type = "primesense"
        cam_filename = "camera_{}.json".format(cam_type)

    elif dataset_name == "ycbv":
        # Includes images captured by two sensors. Use the "UW" sensor as default.
        if cam_type is None:
            cam_type = "uw"
        cam_filename = "camera_{}.json".format(cam_type)

    else:
        cam_filename = "camera.json"

    # Path to the camera file.
    cam_params_path = join(datasets_path, dataset_name, cam_filename)

    p = {
        # Path to a file with camera parameters.
        "cam_params_path": cam_params_path,
    }

    # Add a dictionary containing the intrinsic camera matrix ('K'), image size
    # ('im_size'), and scale of the depth images ('depth_scale', optional).
    p.update(inout.load_cam_params(cam_params_path))

    return p


def get_model_params(datasets_path, dataset_name, model_type=None):
    """Returns parameters of object models for the specified dataset.

    :param datasets_path: Path to a folder with datasets.
    :param dataset_name: Name of the dataset for which to return the parameters.
    :param model_type: Type of object models.
    :return: Dictionary with object model parameters for the specified dataset.
    """
    # Object ID's.
    obj_ids = {
        "lm": list(range(1, 16)),
        "lmo": [1, 5, 6, 8, 9, 10, 11, 12],
        "tless": list(range(1, 31)),
        "tudl": list(range(1, 4)),
        "tyol": list(range(1, 22)),
        "ruapc": list(range(1, 15)),
        "icmi": list(range(1, 7)),
        "icbin": list(range(1, 3)),
        "itodd": list(range(1, 29)),
        "hbs": [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33],
        "hb": list(range(1, 34)),  # Full HB dataset.
        "ycbv": list(range(1, 22)),
        "hope": list(range(1, 29)),
        "hopev2": list(range(1, 29)),
        "hot3d": list(range(1, 34)),
        "handal": list(range(1, 41)),
        "robi": list(range(1, 8)),
    }[dataset_name]

    # ID's of objects with ambiguous views evaluated using the ADI pose error
    # function (the others are evaluated using ADD). See Hodan et al. (ECCVW'16).
    symmetric_obj_ids = {
        "lm": [3, 7, 10, 11],
        "lmo": [10, 11],
        "tless": list(range(1, 31)),
        "tudl": [],
        "tyol": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21],
        "ruapc": [8, 9, 12, 13],
        "icmi": [1, 2, 6],
        "icbin": [1],
        "itodd": [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 24, 25, 27, 28],
        "hbs": [10, 12, 18, 29],
        "hb": [6, 10, 11, 12, 13, 14, 18, 24, 29],
        "ycbv": [1, 13, 14, 16, 18, 19, 20, 21],
        "hope": [],
        "hopev2": [],
        "hot3d": [1, 2, 3, 5, 22, 24, 25, 29, 30, 32],
        "handal": [26, 35, 36, 37, 38, 39, 40],
        "robi": [1, 2, 4, 7],
    }[dataset_name]

    # T-LESS includes two types of object models, CAD and reconstructed.
    # Use the CAD models as default.
    if dataset_name == "tless" and model_type is None:
        model_type = "cad"

    # Both versions of the HB dataset share the same directory.
    if dataset_name == "hbs":
        dataset_name = "hb"

    # Name of the folder with object models.
    models_folder_name = "models"
    if model_type is not None:
        models_folder_name += "_" + model_type

    # Path to the folder with object models.
    models_path = join(datasets_path, dataset_name, models_folder_name)

    p = {
        # ID's of all objects included in the dataset.
        "obj_ids": obj_ids,
        # ID's of objects with symmetries.
        "symmetric_obj_ids": symmetric_obj_ids,
        # Path template to an object model file.
        "model_tpath": join(models_path, "obj_{obj_id:06d}.ply"),
        # Path to a file with meta information about the object models.
        "models_info_path": join(models_path, "models_info.json"),
    }

    return p


def get_split_params(datasets_path, dataset_name, split, split_type=None):
    """Returns parameters (camera params, paths etc.) for the specified dataset.

    :param datasets_path: Path to a folder with datasets.
    :param dataset_name: Name of the dataset for which to return the parameters.
    :param split: Name of the dataset split ('train', 'val', 'test').
    :param split_type: Name of the split type (e.g. for T-LESS, possible types of
      the 'train' split are: 'primesense', 'render_reconst').
    :return: Dictionary with parameters for the specified dataset split.
    """
    p = {
        "name": dataset_name,
        "split": split,
        "split_type": split_type,
        "base_path": join(datasets_path, dataset_name),
        "depth_range": None,
        "azimuth_range": None,
        "elev_range": None,
    }

    rgb_ext = ".png"
    gray_ext = ".png"
    depth_ext = ".png"

    if split_type == "pbr":
        # The photorealistic synthetic images are provided in the JPG format.
        rgb_ext = ".jpg"
    elif dataset_name == "itodd":
        gray_ext = ".tif"
        depth_ext = ".tif"

    p["im_modalities"] = ["rgb", "depth"]
    # for Classic datasets, test modality is implicit...
    p["eval_modality"] = None
    # ...and only one set of annotation is present in the dataset
    # (e.g. scene_gt.json instead of scene_gt_rgb.json, scene_gt_gray1.json etc.)
    modalities_have_separate_annotations = False 
    exts = None  # has to be set if modalities_have_separate_annotations is True

    supported_error_types = ["ad", "add", "adi", "vsd", "mssd", "mspd", "cus", "proj"]

    # Linemod (LM).
    if dataset_name == "lm":
        p["scene_ids"] = list(range(1, 16))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (600.90, 1102.35)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (0, 0.5 * math.pi)

    # Linemod-Occluded (LM-O).
    elif dataset_name == "lmo":
        p["scene_ids"] = {"train": [1, 5, 6, 8, 9, 10, 11, 12], "test": [2]}[split]
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (346.31, 1499.84)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (0, 0.5 * math.pi)

    # T-LESS.
    elif dataset_name == "tless":
        if split == "train":
            if split_type == "synthetless":
                p["scene_ids"] = [1]
            else:
                p["scene_ids"] = list(range(1, 31))
        elif split == "test":
            p["scene_ids"] = list(range(1, 21))

        # Use images from the Primesense sensor by default.
        if split_type is None:
            split_type = "primesense"

        p["im_size"] = {
            "train": {
                "primesense": (400, 400),
                "kinect": (400, 400),
                "canon": (1900, 1900),
                "render_reconst": (1280, 1024),
                "pbr": (720, 540),
                "synthetless": (400, 400),
            },
            "test": {
                "primesense": (720, 540),
                "kinect": (720, 540),
                "canon": (2560, 1920),
            },
        }[split][split_type]

        # The following holds for Primesense, but is similar for the other sensors.
        if split == "test":
            p["depth_range"] = (649.89, 940.04)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # TU Dresden Light (TUD-L).
    elif dataset_name == "tudl":
        if split == "train" and split_type is None:
            split_type = "render"

        p["scene_ids"] = list(range(1, 4))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (851.29, 2016.14)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.4363, 0.5 * math.pi)  # (-25, 90) [deg].

    # Toyota Light (TYO-L).
    elif dataset_name == "tyol":
        p["scene_ids"] = list(range(1, 22))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (499.57, 1246.07)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # Rutgers APC (RU-APC).
    elif dataset_name == "ruapc":
        p["scene_ids"] = list(range(1, 15))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (594.41, 739.12)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # Tejani et al. (IC-MI).
    elif dataset_name == "icmi":
        p["scene_ids"] = list(range(1, 7))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (509.12, 1120.41)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (0, 0.5 * math.pi)

    # Doumanoglou et al. (IC-BIN).
    elif dataset_name == "icbin":
        p["scene_ids"] = {"train": list(range(1, 3)), "test": list(range(1, 4))}[split]
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (454.56, 1076.29)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-1.0297, 0.5 * math.pi)  # (-59, 90) [deg].

    # MVTec ITODD.
    elif dataset_name == "itodd":
        p["scene_ids"] = {"train": [], "val": [1], "test": [1]}[split]
        p["im_size"] = (1280, 960)

        p["im_modalities"] = ["gray", "depth"]

        if split == "test":
            p["depth_range"] = (638.38, 775.97)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # HomebrewedDB (HB).
    # 'hbs' -- Subset of the HB dataset used in the BOP Challenge 2019/2020.
    # 'hb' -- Full HB dataset.
    elif dataset_name in ["hbs", "hb"]:
        dataset_name_orig = dataset_name
        dataset_name = "hb"

        # Use images from the Primesense sensor by default.
        if split_type is None:
            split_type = "primesense"

        if dataset_name_orig == "hbs":
            p["scene_ids"] = {"train": [], "val": [3, 5, 13], "test": [3, 5, 13]}[split]
        else:
            p["scene_ids"] = {
                "train": [],
                "val": list(range(1, 14)),
                "test": list(range(1, 14)),
            }[split]

        p["im_size"] = {
            "pbr": (640, 480),
            "primesense": (640, 480),
            "kinect": (1920, 1080),
        }[split_type]

        # The following holds for Primesense, but is similar for Kinect.
        if split == "test":
            p["depth_range"] = (438.24, 1416.97)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # YCB-Video (YCBV).
    elif dataset_name == "ycbv":
        if split == "train" and split_type is None:
            split_type = "real"

        if split == "train":
            p["scene_ids"] = {
                "real": list(range(48)) + list(range(60, 92)),
                "pbr": None,  # Use function get_present_scene_ids().
                "synt": list(range(80)),
            }[split_type]
        elif split == "test":
            p["scene_ids"] = list(range(48, 60))

        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (612.92, 1243.59)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-1.2788, 1.1291)  # (-73.27, 64.69) [deg].

    # HOPE.
    elif dataset_name == "hope":
        p["scene_ids"] = {
            "train": [],
            "val": list(range(1, 11)),
            "test": list(range(1, 41)),
        }[split]
        p["im_size"] = (1920, 1080)

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

    # HOPEV2.
    elif dataset_name == "hopev2":
        p["scene_ids"] = {
            "train": [],
            "val": list(range(1, 11)),
            "test": list(range(1, 48)),
        }[split]
        p["im_size"] = (1920, 1080)

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

    # HANDAL.
    elif dataset_name == "handal":
        p["scene_ids"] = {
            "train": [],
            "val": list(range(1, 11)),
            "test": list(range(11, 72)),
        }[split]
        p["im_size"] = (1920, 1440)

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

    # HOT3D.
    elif dataset_name == "hot3d":
        modalities_have_separate_annotations = True 
        p["im_modalities"] = ["rgb","gray1","gray2"]
        p["test_quest3_scene_ids"] = list(range(1288, 1849))
        p["test_aria_scene_ids"] = list(range(3365, 3832))
        p["train_quest3_scene_ids"] = list(range(0, 1288))
        p["train_aria_scene_ids"] = list(range(1849, 3365))
        p["scene_ids"] = {
            "test": p["test_quest3_scene_ids"] + p["test_aria_scene_ids"],  # test_quest3 + test_aria
            "train": p["train_quest3_scene_ids"] + p["train_aria_scene_ids"],  # train_quest3 + train_aria
        }[split]
        p["quest3_im_size"] = {"gray1": (1280, 1024), "gray2": (1280, 1024)}
        p["aria_im_size"] = {"rgb": (1408, 1408), "gray1": (640, 480), "gray2": (640, 480)}

        p["quest3_eval_modality"] = "gray1"
        p["aria_eval_modality"] = "rgb"
        def hot3d_eval_modality(scene_id):
            if scene_id in p["test_quest3_scene_ids"] or scene_id in p["train_quest3_scene_ids"]:
                return p["quest3_eval_modality"]
            elif scene_id in p["test_aria_scene_ids"] or scene_id in p["train_aria_scene_ids"]:
                return p["aria_eval_modality"]
            else:
                raise ValueError("scene_id {} not part of hot3d valid scenes".format(scene_id))

        p["eval_modality"] = hot3d_eval_modality

        exts = {
            "rgb": ".jpg",
            "gray1": ".jpg",
            "gray2": "jpg",
        }

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

        supported_error_types = ["ad", "add", "adi", "mssd", "mspd"]

    elif dataset_name[:4] == "robi":
        rgb_ext = ".png"
        # easy to mask mistake, use get_present_scene_ids instead
        base_path = join(datasets_path, dataset_name)
        split_path = join(base_path, split)
        if split_type is not None:
            split_path += "_" + split_type
        p["split_path"] = split_path
        p["scene_ids"] = get_present_scene_ids(dp_split=p)
        p["im_size"] = {
            "realsense": (1280, 720),
            "ensenso": (1280, 1024),
        }.get(split_type, (1280, 1024))
        if split_type in ["realsense", "ensenso"]:
            rgb_ext = ".bmp"
    else:
        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))

    base_path = join(datasets_path, dataset_name)
    split_path = join(base_path, split)
    if split_type is not None:
        if split_type == "pbr" and dataset_name not in ["robi"]:
            p["scene_ids"] = list(range(50))
        split_path += "_" + split_type

    # Path to the split directory.
    p["split_path"] = split_path
    p["supported_error_types"] = supported_error_types
    if not modalities_have_separate_annotations:
        p.update(
            {
                # Path template to a gray image.
                "gray_tpath": join(
                    split_path, "{scene_id:06d}", "gray", "{im_id:06d}" + gray_ext
                ),
                # Path template to an RGB image.
                "rgb_tpath": join(
                    split_path, "{scene_id:06d}", "rgb", "{im_id:06d}" + rgb_ext
                ),
                # Path template to a depth image.
                "depth_tpath": join(
                    split_path, "{scene_id:06d}", "depth", "{im_id:06d}" + depth_ext
                ),
                # Path template to a file with per-image camera parameters.
                "scene_camera_tpath": join(
                    split_path, "{scene_id:06d}", "scene_camera.json"
                ),
                # Path template to a file with GT annotations.
                "scene_gt_tpath": join(
                    split_path, "{scene_id:06d}", "scene_gt.json"
                ),
                # Path template to a file with meta information about the GT annotations.
                "scene_gt_info_tpath": join(
                    split_path, "{scene_id:06d}", "scene_gt_info.json"
                ),
                # Path template to a file with the coco GT annotations.
                "scene_gt_coco_tpath": join(
                    split_path, "{scene_id:06d}", "scene_gt_coco.json"
                ),
                # Path template to a mask of the full object silhouette.
                "mask_tpath": join(
                    split_path, "{scene_id:06d}", "mask", "{im_id:06d}_{gt_id:06d}.png"
                ),
                # Path template to a mask of the visible part of an object silhouette.
                "mask_visib_tpath": join(
                    split_path,
                    "{scene_id:06d}",
                    "mask_visib",
                    "{im_id:06d}_{gt_id:06d}.png",
                ),
            }
        )

    else:
        assert exts is not None, "Need to set 'exts' for dataset {}".format()
        for moda in p["im_modalities"]:
            p.update(
                {
                    # Path template to modality image.
                    "{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", moda, "{im_id:06d}" + exts[moda]
                    ),
                    # Path template to a file with per-image camera parameters.
                    "scene_camera_{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", "scene_camera_{}.json".format(moda)
                    ),
                    # Path template to a file with GT annotations.
                    "scene_gt_{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", "scene_gt_{}.json".format(moda)
                    ),
                    # Path template to a file with meta information about the GT annotations.
                    "scene_gt_info_{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", "scene_gt_info_{}.json".format(moda)
                    ),
                    # Path template to a file with the coco GT annotations.
                    "scene_gt_coco_{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", "scene_gt_coco_{}.json".format(moda)
                    ),
                    # Path template to a mask of the full object silhouette.
                    "mask_{}_tpath".format(moda): join(
                        split_path, "{scene_id:06d}", "mask_{}".format(moda), "{im_id:06d}_{gt_id:06d}.png"
                    ),
                    # Path template to a mask of the visible part of an object silhouette.
                    "mask_visib_{}_tpath".format(moda): join(
                        split_path,
                        "{scene_id:06d}",
                        "mask_visib_{}".format(moda),
                        "{im_id:06d}_{gt_id:06d}.png",
                    ),
                }
            )

    return p


def scene_tpaths_keys(eval_modality, scene_id=None):
    """
    Define keys corresponding template path defined in get_split_params output.
    
    Definition for scene gt, scene gt info and scene camera.
    - Classic datasets: "scene_gt_tpath", "scene_gt_info_tpath", "scene_camera_tpath"
    - H3 datasets: with separate annotations for modalities, e.g. "scene_gt_{modality}_tpath", 
    "scene_gt_info_{modality}_tpath", "scene_camera_{modality}_tpath", etc.
    Modality may be the same for the whole dataset split (defined as a `str`), 
    or vary scene by scene (defined as function or a dictionary)

    :param eval_modality: None, str, callable or ditc, defines
    :param scene_id: None or int, should be specified if eval modality 
                     changes from scene to scen
    :return: scene tpath keys dictionary
    """

    tpath_keys = [
        "scene_gt_tpath", "scene_gt_info_tpath", "scene_camera_tpath", 
        "scene_gt_coco_tpath", "mask_tpath", "mask_visib_tpath"
    ]
    tpath_keys_multi = [
        "scene_gt_{}_tpath", "scene_gt_info_{}_tpath", "scene_camera_{}_tpath", 
        "scene_gt_coco_{}_tpath", "mask_{}_tpath", "mask_visib_{}_tpath"
    ]

    assert len(tpath_keys) == len(tpath_keys_multi)
    tpath_keys_dic = {}
    for key, key_multi in zip(tpath_keys, tpath_keys_multi):
        if eval_modality is None:
            # Classic filenames
            tpath_keys_dic[key] = key
        elif isinstance(eval_modality, str):
            tpath_keys_dic[key] = key_multi.format(eval_modality)
        elif callable(eval_modality) and scene_id is not None:
            tpath_keys_dic[key] = key_multi.format(eval_modality(scene_id))
        elif isinstance(eval_modality, dict) and scene_id is not None:
            tpath_keys_dic[key] = key_multi.format(eval_modality[scene_id])
        else:
            raise ValueError("eval_modality type not supported, either None, str, callable or dictionary")
    
    return tpath_keys_dic


def get_present_scene_ids(dp_split):
    """Returns ID's of scenes present in the specified dataset split.

    :param dp_split: Path to a folder with datasets.
    :return: List with scene ID's.
    """
    scene_dirs = [
        d
        for d in glob.glob(os.path.join(dp_split["split_path"], "*"))
        if os.path.isdir(d)
    ]
    scene_ids = [int(os.path.basename(scene_dir)) for scene_dir in scene_dirs]
    scene_ids = sorted(scene_ids)
    return scene_ids
