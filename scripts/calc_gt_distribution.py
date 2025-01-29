# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates distribution of GT poses."""
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "ycbv",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Modality used to compute gt statistics, defaults to eval modality
    "modality": None,
    # Sensor used to compute gt statistics, defaults to eval sensor
    "sensor": None,
    # Folder for output visualisations.
    "vis_path": os.path.join(config.output_path, "gt_distribution"),
    # Save plots in "vis_path"
    "save_plots": True,
    # Show plots"
    "show_plots": True,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
)

if p["modality"] is None:
    p["modality"] = dp_split["eval_modality"]
if p["sensor"] is None:
    p["sensor"] = dp_split["eval_sensor"]

scene_ids = dp_split["scene_ids"]
dists = []
azimuths = []
elevs = []
visib_fracts = []
ims_count = 0

for scene_id in scene_ids:
    tpath_keys = dataset_params.scene_tpaths_keys(p["modality"], p["sensor"], scene_id)

    misc.log(f"Processing - dataset: {p['dataset']} ({p['dataset_split']}, {p['dataset_split_type']}), scene: {scene_id}")

    # Load GT poses.
    scene_gt_path = dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
    scene_gt = inout.load_scene_gt(scene_gt_path)
    
    # Load info about the GT poses.
    scene_gt_info_path = dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
    scene_gt_info = inout.load_json(scene_gt_info_path, keys_to_int=True)

    ims_count += len(scene_gt)

    for im_id in scene_gt:
        for gt_id, im_gt in enumerate(scene_gt[im_id]):
            # Object distance.
            dist = np.linalg.norm(im_gt["cam_t_m2c"])
            dists.append(dist)

            # Camera origin in the model coordinate system.
            cam_orig_m = -np.linalg.inv(im_gt["cam_R_m2c"]).dot(im_gt["cam_t_m2c"])

            # Azimuth from [0, 360].
            azimuth = math.atan2(cam_orig_m[1, 0], cam_orig_m[0, 0])
            if azimuth < 0:
                azimuth += 2.0 * math.pi
            azimuths.append((180.0 / math.pi) * azimuth)

            # Elevation from [-90, 90].
            a = np.linalg.norm(cam_orig_m)
            b = np.linalg.norm([cam_orig_m[0, 0], cam_orig_m[1, 0], 0])
            elev = math.acos(b / a)
            if cam_orig_m[2, 0] < 0:
                elev = -elev
            elevs.append((180.0 / math.pi) * elev)

            # Visibility fraction.
            visib_fracts.append(scene_gt_info[im_id][gt_id]["visib_fract"])

# Print stats.
misc.log(
    "Stats of the GT poses in dataset {} {}:".format(p["dataset"], p["dataset_split"])
)
misc.log("Number of images: " + str(ims_count))

if ims_count == 0:
    misc.log("No ground truth found.")
    exit()

misc.log("Min dist: {}".format(np.min(dists)))
misc.log("Max dist: {}".format(np.max(dists)))
misc.log("Mean dist: {}".format(np.mean(dists)))

misc.log("Min azimuth: {}".format(np.min(azimuths)))
misc.log("Max azimuth: {}".format(np.max(azimuths)))
misc.log("Mean azimuth: {}".format(np.mean(azimuths)))

misc.log("Min elev: {}".format(np.min(elevs)))
misc.log("Max elev: {}".format(np.max(elevs)))
misc.log("Mean elev: {}".format(np.mean(elevs)))

misc.log("Min visib fract: {}".format(np.min(visib_fracts)))
misc.log("Max visib fract: {}".format(np.max(visib_fracts)))
misc.log("Mean visib fract: {}".format(np.mean(visib_fracts)))

prefix = f"{p['modality']}_{p['sensor']}_" if isinstance(p["modality"], str) else ""
# Visualize distributions.
if p["save_plots"]:
    save_dir = os.path.join(p["vis_path"], p["dataset"])
    misc.log(f"Saving plots in {save_dir}")
    misc.ensure_dir(save_dir)

plt.figure()
plt.hist(dists, bins=100)
plt.title("Object distance")
if p["save_plots"]:
    path = os.path.join(save_dir, f"{prefix}object_distance.png")
    misc.log(f"Saving {path}")
    plt.savefig(path)

plt.figure()
plt.hist(azimuths, bins=100)
plt.title("Azimuth")
if p["save_plots"]:
    path = os.path.join(save_dir, f"{prefix}azimuth.png")
    misc.log(f"Saving {path}")
    plt.savefig(path)

plt.figure()
plt.hist(elevs, bins=100)
plt.title("Elevation")
if p["save_plots"]:
    path = os.path.join(save_dir, f"{prefix}elevation.png")
    misc.log(f"Saving {path}")
    plt.savefig(path)

plt.figure()
plt.hist(visib_fracts, bins=100)
plt.title("Visibility fraction")
if p["save_plots"]:
    path = os.path.join(save_dir, f"{prefix}visibility_fraction.png")
    misc.log(f"Saving {path}")
    plt.savefig(path)

if p["show_plots"]:
    plt.show()
