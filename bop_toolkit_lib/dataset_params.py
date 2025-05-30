# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Parameters of the BOP datasets."""

import math
import glob
import os
from os.path import join
from collections.abc import Callable
from typing import Union, Dict

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

    # hot3d does not have a single camera file, raise an exception
    elif dataset_name in ['hot3d']:
        raise ValueError("BOP dataset {} does not have a global camera file.".format(dataset_name))

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
        "itoddmv": list(range(1, 29)),
        "hbs": [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33],
        "hb": list(range(1, 34)),  # Full HB dataset.
        "ycbv": list(range(1, 22)),
        "hope": list(range(1, 29)),
        "hopev2": list(range(1, 29)),
        "hot3d": list(range(1, 34)),
        "handal": list(range(1, 41)),
        "ipd": [0, 1, 4, 8, 10, 11, 14, 18, 19, 20],
        "xyzibd": [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
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
        "itoddmv": [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 24, 25, 27, 28],
        "hbs": [10, 12, 18, 29],
        "hb": [6, 10, 11, 12, 13, 14, 18, 24, 29],
        "ycbv": [1, 13, 14, 16, 18, 19, 20, 21],
        "hope": [],
        "hopev2": [],
        "hot3d": [1, 2, 3, 5, 22, 24, 25, 29, 30, 32],
        "handal": [26, 35, 36, 37, 38, 39, 40],
        "ipd": [8, 14, 18, 19, 20],
        "xyzibd": [1, 2, 5, 8, 9, 11, 12, 16, 17]
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
    # for Classic datasets, sensor and modality used for the evaluation is implicit...
    p["eval_sensor"] = None
    p["eval_modality"] = None
    # ...and only one set of annotation is present in the dataset
    # (e.g. scene_gt.json instead of scene_gt_rgb.json, scene_gt_gray1.json etc.)
    sensor_modalities_have_separate_annotations = False
    # file extensions for datasets with multiple sensor/modalities options
    # has to be set if sensor_modalities_have_separate_annotations is True
    exts = None

    supported_error_types = ["ad", "add", "adi", "vsd", "mssd", "mspd", "cus", "proj"]

    # Linemod (LM).
    if dataset_name == "lm":
        p["scene_ids"] = list(range(1, 16))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (600.90, 1102.35)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (0, 0.5 * math.pi)

    # Linemod-Occluded (LM-O).
    elif dataset_name == "lmo":
        p["scene_ids"] = {"train": [1, 5, 6, 8, 9, 10, 11, 12], "test": [2]}[split]
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (346.31, 1499.84)  # Range of camera-object distances.
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
            p["depth_range"] = (649.89, 940.04)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # TU Dresden Light (TUD-L).
    elif dataset_name == "tudl":
        if split == "train" and split_type is None:
            split_type = "render"

        p["scene_ids"] = list(range(1, 4))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (569.88, 1995.27)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.4363, 0.5 * math.pi)  # (-25, 90) [deg].

    # Toyota Light (TYO-L).
    elif dataset_name == "tyol":
        p["scene_ids"] = list(range(1, 22))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (499.57, 1246.07)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # Rutgers APC (RU-APC).
    elif dataset_name == "ruapc":
        p["scene_ids"] = list(range(1, 15))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (594.41, 739.12)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    # Tejani et al. (IC-MI).
    elif dataset_name == "icmi":
        p["scene_ids"] = list(range(1, 7))
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (509.12, 1120.41)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (0, 0.5 * math.pi)

    # Doumanoglou et al. (IC-BIN).
    elif dataset_name == "icbin":
        p["scene_ids"] = {"train": list(range(1, 3)), "test": list(range(1, 4))}[split]
        p["im_size"] = (640, 480)

        if split == "test":
            p["depth_range"] = (454.56, 1076.29)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-1.0297, 0.5 * math.pi)  # (-59, 90) [deg].

    # MVTec ITODD.
    elif dataset_name == "itodd":
        p["scene_ids"] = {"train": [], "val": [1], "test": [1]}[split]
        p["im_size"] = (1280, 960)

        p["im_modalities"] = ["gray", "depth"]

        if split == "test":
            p["depth_range"] = (638.38, 775.97)  # Range of camera-object distances.
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
            p["depth_range"] = (438.24, 1416.97)  # Range of camera-object distances.
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
            p["depth_range"] = (612.92, 1243.59)  # Range of camera-object distances.
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
        sensor_modalities_have_separate_annotations = {"aria": True, "quest3": True}
        p["im_modalities"] = {"aria": ["rgb", "gray1", "gray2"], "quest3": ["gray1", "gray2"]}
        p["test_quest3_scene_ids"] = list(range(1288, 1849))
        p["test_aria_scene_ids"] = list(range(3365, 3832))
        p["train_quest3_scene_ids"] = list(range(0, 1288))
        p["train_aria_scene_ids"] = list(range(1849, 3365))
        p["scene_ids"] = {
            "test": p["test_quest3_scene_ids"] + p["test_aria_scene_ids"],  # test_quest3 + test_aria
            "train": p["train_quest3_scene_ids"] + p["train_aria_scene_ids"],  # train_quest3 + train_aria
        }[split]

        p["im_size"] = {
            "aria" : {"rgb": (1408, 1408), "gray1": (640, 480), "gray2": (640, 480)},
            "quest3" : {"gray1": (1280, 1024), "gray2": (1280, 1024)}
        }

        p["quest3_eval_modality"] = "gray1"
        p["aria_eval_modality"] = "rgb"
        def hot3d_eval_modality(scene_id):
            if scene_id in p["test_quest3_scene_ids"] or scene_id in p["train_quest3_scene_ids"]:
                return p["quest3_eval_modality"]
            elif scene_id in p["test_aria_scene_ids"] or scene_id in p["train_aria_scene_ids"]:
                return p["aria_eval_modality"]
            else:
                raise ValueError("scene_id {} not part of hot3d valid scenes".format(scene_id))

        def hot3d_eval_sensor(scene_id):
            if scene_id in p["test_quest3_scene_ids"] or scene_id in p["train_quest3_scene_ids"]:
                return "quest3"
            elif scene_id in p["test_aria_scene_ids"] or scene_id in p["train_aria_scene_ids"]:
                return "aria"
            else:
                raise ValueError("scene_id {} not part of hot3d valid scenes".format(scene_id))

        p["eval_modality"] = hot3d_eval_modality
        p["eval_sensor"] = hot3d_eval_sensor

        exts = {
            "aria" : {"rgb": ".jpg", "gray1": ".jpg", "gray2": ".jpg"},
            "quest3": {"gray1": ".jpg", "gray2": ".jpg"}
        }

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

        supported_error_types = ["ad", "add", "adi", "mssd", "mspd"]
    elif dataset_name == "ipd":
            sensor_modalities_have_separate_annotations = {"photoneo": False, "cam1" : False, "cam2" : False, "cam3" : False}
            p["im_modalities"] = {"photoneo": ["rgb", "depth"], "cam1" : ["rgb", "aolp", "dolp", "depth"],
                                  "cam2" : ["rgb", "aolp", "dolp", "depth"], "cam3" : ["rgb", "aolp", "dolp", "depth"]}
            p["scene_ids"] = {
                "test": list(range(15)),
                "train": list(range(10)),
                "val": list(range(15)),
            }[split]

            p["im_size"] = {
                "photoneo" : (2064, 1544),
                "cam1" : (3840, 2160),
                "cam2": (3840, 2160),
                "cam3": (3840, 2160),
                "": (2400, 2400),
            }

            p["eval_modality"] = "rgb"
            p["eval_sensor"] = "photoneo"

            exts = {
                "photoneo": {"rgb": ".png", "depth": ".png"},
                "cam1": {"rgb": ".png", "depth": ".png", "aolp": ".png", "dolp": ".png"},
                "cam2": {"rgb": ".png", "depth": ".png", "aolp": ".png", "dolp": ".png"},
                "cam3": {"rgb": ".png", "depth": ".png", "aolp": ".png", "dolp": ".png"},
            }

            if split == "test":
                p["depth_range"] = None  # Not calculated yet.
                p["azimuth_range"] = None  # Not calculated yet.
                p["elev_range"] = None  # Not calculated yet.

            supported_error_types = ["ad", "add", "adi", "mssd", "mspd"]

    elif dataset_name == "xyzibd":
        sensor_modalities_have_separate_annotations = {"photoneo": False, "xyz": False, "realsense": False}
        p["im_modalities"] = {"photoneo": ["gray", "depth"], "xyz": ["gray", "depth"], "realsense": ["rgb", "depth"]}
        val_scene_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 54, 60, 65, 70]
        missing_pbr_scene_ids = [6, 7, 8, 18, 19, 20]
        p["scene_ids"] = {
            "test": [i for i in range(1, 75) if i not in val_scene_ids],
            "val": val_scene_ids,
            "train": [i for i in list(range(51)) if i not in missing_pbr_scene_ids],
        }[split]

        p["im_size"] = {
            "xyz": (1440, 1080),
            "realsense": (1280, 720),
            "photoneo": (2064, 1544),
            "": (1440, 1080),
        }

        p["eval_modality"] = "gray"
        p["eval_sensor"] = "xyz"

        if "pbr" == split_type:
            # The PBR data is in classical BOP format without sensor names.
            p["eval_modality"] = None
            p["eval_sensor"] = None
            sensor_modalities_have_separate_annotations = False

        exts = {
            "photoneo": {"gray": ".png", "depth": ".png"},
            "xyz": {"gray": ".png", "depth": ".png"},
            "realsense": {"rgb": ".png", "depth": ".png"},
        }

        if split == "test":
            p["depth_range"] = None  # Not calculated yet.
            p["azimuth_range"] = None  # Not calculated yet.
            p["elev_range"] = None  # Not calculated yet.

        supported_error_types = ["ad", "add", "adi", "mssd", "mspd"]
    elif dataset_name == "itoddmv":
        sensor_modalities_have_separate_annotations = {"3dlong": False, "cam0": False, "cam1": False, "cam2": False}
        p["im_modalities"] = {"3dlong": ["gray", "depth"], "cam0": ["gray"], "cam1": ["gray"], "cam2": ["gray"]}
        p["scene_ids"] = {
            "test": [1],
            "train": list(range(50)),
        }[split]

        p["im_size"] = {
            "3dlong": (1280, 960),
            "cam0": (4224, 2838),
            "cam1": (4224, 2838),
            "cam2": (4224, 2838),
            "": (1280, 960),
        }

        p["eval_modality"] = "gray"
        p["eval_sensor"] = "3dlong"

        if "pbr" == split_type:
            # The PBR data is in classical BOP format without sensor names.
            p["eval_modality"] = None
            p["eval_sensor"] = None
            sensor_modalities_have_separate_annotations = False

        exts = {
            "3dlong": {"gray": ".tif", "depth": ".tif"},
            "cam0": {"gray": ".tif"},
            "cam1": {"gray": ".tif"},
            "cam2": {"gray": ".tif"},
        }

        if split == "test":
            p["depth_range"] = (638.38, 775.97)  # Range of camera-object distances.
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

        supported_error_types = ["ad", "add", "adi", "mssd", "mspd"]

    else:
        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))

    base_path = join(datasets_path, dataset_name)
    split_path = join(base_path, split)
    if split_type is not None:
        if split_type == "pbr" and dataset_name != "xyzibd":
            p["scene_ids"] = list(range(50))
        split_path += "_" + split_type

    # Path to the split directory.
    p["split_path"] = split_path
    p["supported_error_types"] = supported_error_types

    # For classic BOP format datasets with one gt file per folder
    classic_bop_format = type(p["im_modalities"]) is list
    if classic_bop_format:
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
        # im_modalities is a dict from sensor to modalities
        for sensor, modalities in p["im_modalities"].items():
            for modality in modalities:
                # If modalities have aligned extrinsics/intrinsics they are combined in one file
                gt_file_suffix = sensor
                # If modalities have separate extrinsics/intrinsics they are accessed by unique modalities (compatible with hot3d)
                if sensor_modalities_have_separate_annotations[sensor]:
                    gt_file_suffix = modality

                # Path template to modality image.
                if dataset_name == "hot3d":
                    p[f"{modality}_{sensor}_tpath"] = join(
                        split_path, "{scene_id:06d}", f"{modality}", "{im_id:06d}" + exts[sensor][modality]
                    )
                else:
                    p[f"{modality}_{sensor}_tpath"] = join(
                        split_path, "{scene_id:06d}", f"{modality}_{sensor}", "{im_id:06d}" + exts[sensor][modality]
                    )
                p.update(
                    {
                        # Path template to a file with per-image camera parameters.
                        "scene_camera_{}_{}_tpath".format(modality, sensor): join(
                            split_path, "{scene_id:06d}", "scene_camera_{}.json".format(gt_file_suffix)
                        ),
                        # Path template to a file with GT annotations.
                        "scene_gt_{}_{}_tpath".format(modality, sensor): join(
                            split_path, "{scene_id:06d}", "scene_gt_{}.json".format(gt_file_suffix)
                        ),
                        # Path template to a file with meta information about the GT annotations.
                        "scene_gt_info_{}_{}_tpath".format(modality, sensor): join(
                            split_path, "{scene_id:06d}", "scene_gt_info_{}.json".format(gt_file_suffix)
                        ),
                        # Path template to a file with the coco GT annotations.
                        "scene_gt_coco_{}_{}_tpath".format(modality, sensor): join(
                            split_path, "{scene_id:06d}", "scene_gt_coco_{}.json".format(gt_file_suffix)
                        ),
                        # Path template to a mask of the full object silhouette.
                        "mask_{}_{}_tpath".format(modality, sensor): join(
                            split_path, "{scene_id:06d}", "mask_{}".format(gt_file_suffix), "{im_id:06d}_{gt_id:06d}.png"
                        ),
                        # Path template to a mask of the visible part of an object silhouette.
                        "mask_visib_{}_{}_tpath".format(modality, sensor): join(
                            split_path,
                            "{scene_id:06d}",
                            "mask_visib_{}".format(gt_file_suffix),
                            "{im_id:06d}_{gt_id:06d}.png",
                        ),
                    }
                )

    return p


def get_scene_sensor_or_modality(
        sm: Union[None, str, Callable],
        scene_id: Union[None, int]
    ) -> Union[None,str]:
    """
    Get sensor|modality associated with a given scene.

    Some datasets (hot3d) have different sensor|modality available depending on the scene.
    Same logic for sensor or modality.
    """
    if sm is None or isinstance(sm, str):
        return sm
    elif callable(sm):
        return sm(scene_id)
    else:
        raise TypeError(f"Sensor or modality {sm} should be either None, str or callable, not {type(sm)}")


def scene_tpaths_keys(
        modality: Union[None, str, Callable],
        sensor: Union[None, str, Callable],
        scene_id: Union[None, int] = None
    ) -> Dict[str,str]:
    """
    Define keys corresponding template path defined in get_split_params output.

    Definition for scene gt, scene gt info and scene camera.
    - Classic datasets (handal and hopev2 included): "scene_gt_tpath", "scene_gt_info_tpath", "scene_camera_tpath", etc.
    - hot3d and Industrial datasets: same tpath keys with modality and sensor,
    e.g. "scene_gt_{modality}_{sensor}_tpath", "scene_gt_info_{modality}_{sensor}_tpath",
    "scene_camera_{modality}_{sensor}_tpath", etc.
    Modality|sensor may be the same for the whole dataset split (defined as a `str`),
    or vary scene by scene (defined as function).

    :param modality: None, str or callable
    :param sensor: None, str or callable
    :param scene_id: None or int, should be specified if eval modality|sensor
                     changes from scene to scene
    :return: scene tpath keys dictionary
    """

    scene_sensor = get_scene_sensor_or_modality(sensor, scene_id)
    scene_modality = get_scene_sensor_or_modality(modality, scene_id)

    # 2 valid combinations:
    # - modality and sensor are None -> BOP classic format
    # - modality and sensor are not None -> hot3d + BOP industrial format
    assert ((scene_modality is None and scene_sensor is None) or (scene_modality is not None and scene_sensor is not None)), f"scene_modality={scene_modality}, scene_sensor={scene_sensor}"

    # "rgb_tpath" refers to the template path key of the given modality|sensor pair
    tpath_keys = [
        "scene_gt_tpath", "scene_gt_info_tpath", "scene_camera_tpath",
        "scene_gt_coco_tpath", "mask_tpath", "mask_visib_tpath", "rgb_tpath"
    ]
    tpath_keys_multi = [
        "scene_gt_{}_{}_tpath", "scene_gt_info_{}_{}_tpath", "scene_camera_{}_{}_tpath",
        "scene_gt_coco_{}_{}_tpath", "mask_{}_{}_tpath", "mask_visib_{}_{}_tpath", "{}_{}_tpath"
    ]
    assert len(tpath_keys) == len(tpath_keys_multi)

    tpath_keys_dic = {}
    for key, key_multi in zip(tpath_keys, tpath_keys_multi):
        if scene_sensor is None:
            # BOP-Classic filenames
            tpath_keys_dic[key] = key
        else:
            tpath_keys_dic[key] = key_multi.format(scene_modality, scene_sensor)

    tpath_keys_dic["depth_tpath"] = tpath_keys_dic["rgb_tpath"].replace("rgb","depth").replace("gray","depth")
    return tpath_keys_dic


def sensor_has_modality(dp_split: Dict, sensor: str, modality: str):
    if isinstance(dp_split["im_modalities"], list):
        return modality in dp_split["im_modalities"]
    else:
        return modality in dp_split["im_modalities"][sensor]


def get_im_size(dp_split: Dict, modality: str, sensor: str):
    """
    Conveniance function to retrieve the image size of a modality|sensor pair.
    """
    if isinstance(dp_split["im_size"], dict):
        if isinstance(dp_split["im_size"][sensor], dict):
            # hot3d
            return dp_split["im_size"][sensor][modality]
        else:
            # BOP Industrial
            return dp_split["im_size"][sensor]
    # BOP Classic: one image size for the whole dataset
    else:
        return dp_split["im_size"]


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
