# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Script to subsample test set from the original dataset."""
# use: python scripts/subsample_dataset.py --dataset_dir ./lm/test --num_scene_images 10

import os
import time
import argparse
import numpy as np
import math
from scipy.cluster.vq import kmeans

from bop_toolkit_lib import inout
from bop_toolkit_lib import transform

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]

# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    "targets_filename": "test_targets_bop19.json",
}
# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    required=True,
    help="Path to the dataset directory containing the scenes: 00000, 000001, etc.",
)
parser.add_argument(
    "--num_scene_images",
    type=int,
    help="Number of images to subsample from each scene.",
)
args = parser.parse_args()
p["dataset_dir"] = str(args.dataset_dir)


def quat_geo_dist(q1, q2):
    """Geodesic distance of quaternions representing 3D rotations."""
    assert q1.size == q2.size == 4
    return 2 * math.acos(np.abs(q1.flatten().dot(q2.flatten())))


def closest_centroid(points, centroids):
    """Finds the index to the nearest centroid for each point."""
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def subsample_images(num_images, scene_gt, scene_gt_info):
    """
    Subsamples a given number of images from the scene_gt and scene_gt_info
    For each object instance, a descriptor is computed from the 3D rotation, translation and visibility fraction.
    The descriptors are then clustered and the images are selected from the clusters.

    Args:
        num_images: num of images to subsample per scene
        scene_gt: gt of the scene
        scene_gt_info: gt info of the scene

    Returns:
        im_ids_selected: list of image ids to be used for the scene
    """
    assert scene_gt.keys() == scene_gt_info.keys()

    # If there are less than the required number of images, take all.
    if len(scene_gt) <= num_images:
        return range(len(scene_gt))

    # Reference quaternions to which distances will be computed
    ref_quats = np.eye(4)

    # Calculate GT descriptors
    gt_descs_list = []
    gt_src_ims_list = []
    im_ids = [k for k in scene_gt.keys()]
    for im_id in im_ids:
        for gt_id in range(len(scene_gt[im_id])):
            # 3D orientation - calculate the geodesic distance
            # of quaternions to the reference quaternions
            R = np.eye(4)
            R[:3, :3] = np.array(scene_gt[im_id][gt_id]["cam_R_m2c"]).reshape(3, 3)
            q = transform.quaternion_from_matrix(R)
            quat_dists = [quat_geo_dist(q, ref_quats[i]) for i in range(4)]

            # 3D translation
            t = scene_gt[im_id][gt_id]["cam_t_m2c"]

            # GT descriptor and feature weights
            gt_desc = quat_dists + t + [scene_gt_info[im_id][gt_id]["visib_fract"]]

            gt_descs_list.append(gt_desc)
            gt_src_ims_list.append(im_id)

    gt_descs = np.array(gt_descs_list, np.float32)

    # Subtract mean
    gt_descs -= np.mean(gt_descs, axis=0)

    # "Whiten" the image descriptors
    # (the rotation features are scaled with the max std. dev. from the four
    # rotation features; the same is for the translation features)
    gt_descs_std = np.std(gt_descs, axis=0)
    gt_descs_std[:4] = gt_descs_std[:4].max()
    gt_descs_std[4:7] = gt_descs_std[4:7].max()
    gt_descs /= gt_descs_std

    # Calculate image descriptors from the GT descriptors
    im_descs = []
    for im_id in im_ids:
        # Get indices of GT descriptors from the current image
        descs_inds = [
            i for i, src_im_id in enumerate(gt_src_ims_list) if src_im_id == im_id
        ]

        gt_descs_im = gt_descs[np.array(descs_inds), :]
        im_desc = np.concatenate(
            (np.mean(gt_descs_im, axis=0), np.std(gt_descs_im, axis=0))
        )
        im_descs.append(im_desc)

    im_descs = np.array(im_descs)

    # Cluster the descriptors
    codebook, distortion = kmeans(im_descs, num_images)
    print("Distortion: {}".format(distortion))

    # Assign the descriptors to clusters
    im_labels = closest_centroid(im_descs, codebook)

    # From each cluster, pick the descriptor that is furthest from the mean
    # of all descriptors (we want to keep the extrema)
    im_ids_selected = []
    for i in range(num_images):
        descs_inds = np.nonzero(im_labels == i)[0]
        descs = im_descs[descs_inds, :]
        descs_norms = np.linalg.norm(descs, axis=1)
        descs_ind_max = int(np.argmax(descs_norms))
        im_id = int(im_ids[descs_inds[descs_ind_max]])
        im_ids_selected.append(im_id)

    im_ids_selected = sorted(im_ids_selected)
    return im_ids_selected


sample_time_start = time.time()

# Load scene in dataset_dir
scene_ids = [
    scene
    for scene in os.listdir(p["dataset_dir"])
    if os.path.isdir(os.path.join(p["dataset_dir"], scene)) and scene.isdigit()
]
scene_ids = sorted(scene_ids)
assert len(scene_ids) > 0, "No scenes found in the dataset directory."

# formatting test_list following BOP format: im_id, inst_count, obj_id, scene_id
test_list = []
for scene_id in scene_ids:
    print("Processing dataset at {}, scene: {}".format(args.dataset_dir, scene_id))
    scene_gt = inout.load_json(
        os.path.join(args.dataset_dir, scene_id, "scene_gt.json")
    )
    scene_gt_info = inout.load_json(
        os.path.join(args.dataset_dir, scene_id, "scene_gt_info.json")
    )
    im_ids_selected = subsample_images(args.num_scene_images, scene_gt, scene_gt_info)
    im_ids_selected = sorted(im_ids_selected)
    for im_id in im_ids_selected:
        im_gt = scene_gt[str(im_id)]
        im_gt_info = scene_gt_info[str(im_id)]
        # keep instances having visib_factor > 0.0
        selected_inst_ids = [
            inst_id
            for inst_id in range(len(im_gt_info))
            if im_gt_info[inst_id]["visib_fract"] > 0.0
        ]

        inst_per_obj_ids = {}
        for inst_id in range(len(selected_inst_ids)):
            obj_id = im_gt[inst_id]["obj_id"]
            if obj_id not in inst_per_obj_ids:
                inst_per_obj_ids[obj_id] = 1
            else:
                inst_per_obj_ids[obj_id] += 2

        for obj_id in inst_per_obj_ids:
            test_list.append(
                {
                    "im_id": im_id,
                    "inst_count": inst_per_obj_ids[obj_id],
                    "obj_id": obj_id,
                    "scene_id": int(scene_id),
                }
            )

print("Saving the selected image ids...")
out_path = os.path.join(p["dataset_dir"], p["targets_filename"])
inout.save_json(out_path, test_list)
print("Saved at {}".format(out_path))
