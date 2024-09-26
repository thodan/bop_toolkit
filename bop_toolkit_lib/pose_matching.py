# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Matching of estimated poses to the ground-truth poses."""

import numpy as np


def match_poses(errs, error_ths, max_ests_count=0, gt_valid_mask=None):
    """Matches the estimated poses to the ground-truth poses.

    The estimated poses are greedily matched to the ground truth poses in the
    order of decreasing score of the estimates. An estimated pose is matched to a
    ground-truth pose if the error w.r.t. the ground-truth pose is below the
    specified threshold. Each estimated pose is matched to up to one ground-truth
    pose and each ground-truth pose is matched to up to one estimated pose.

    :param errs: List of dictionaries, where each dictionary holds the following
      info about one pose estimate:
      - 'est_id': ID of the pose estimate.
      - 'score': Confidence score of the pose estimate.
      - 'errors': Dictionary mapping ground-truth ID's to errors of the pose
          estimate w.r.t. the ground-truth poses.
    :param error_ths: Thresholds of correctness. The pose error can be given
      by more than one element (e.g. translational + rotational error), in which
      case there is one threshold for each element.
    :param max_ests_count: Top k pose estimates to consider (0 = all).
    :param gt_valid_mask: Mask of ground-truth poses which can be considered.
    :return: List of dictionaries, where each dictionary holds info for one pose
      estimate (the estimates are ordered as in errs) about the matching
      ground-truth pose:
      - 'est_id': ID of the pose estimate.
      - 'gt_id': ID of the matched ground-truth pose (-1 means there is no
          matching ground-truth pose).
      - 'score': Confidence score of the pose estimate.
      - 'error': Error of the pose estimate w.r.t. the matched ground-truth pose.
      - 'error_norm': Error normalized by the threshold value.
    """
    # Sort the estimated poses by decreasing confidence score.
    errs_sorted = sorted(errs, key=lambda e: e["score"], reverse=True)

    # Keep only the required number of poses with the highest confidence score.
    # 0 = all pose estimates are considered.
    if max_ests_count > 0:
        errs_sorted = errs_sorted[:max_ests_count]

    # Number of values defining the error (e.g. 1 for "ADD", 2 for "5deg 5cm").
    error_num_elems = len(list(error_ths))

    # Greedily match the estimated poses to the ground truth poses in the order of
    # decreasing score of the estimates.
    matches = []
    gt_matched = []
    for e in errs_sorted:
        best_gt_id = -1
        best_error = list(error_ths)
        for gt_id, error in e["errors"].items():
            gt_visib_fract = e["gt_visib_fracts"][gt_id]
            # If the mask of valid GT poses is not provided, consider all valid.
            is_valid = not gt_valid_mask or gt_valid_mask[gt_id]

            # Only valid GT poses that have not been matched yet are considered.
            if is_valid and gt_id not in gt_matched:
                # The current pose estimate is considered the best so far if all error
                # elements are the lowest so far.
                if np.all([error[i] < best_error[i] for i in range(error_num_elems)]):
                    best_gt_id = gt_id
                    best_error = error
                    gt_visib_fract_of_best_gt_id = gt_visib_fract

        if best_gt_id >= 0:
            # Mark the GT pose as matched.
            gt_matched.append(best_gt_id)

            # Error normalized by the threshold.
            best_errors_normed = [
                best_error[i] / float(error_ths[i]) for i in range(error_num_elems)
            ]

            # Save info about the match.
            matches.append(
                {
                    "est_id": e["est_id"],
                    "gt_id": best_gt_id,
                    "score": e["score"],
                    "error": best_error,
                    "error_norm": best_errors_normed,
                    "gt_visib_fract": gt_visib_fract_of_best_gt_id,
                }
            )

    return matches


def match_poses_scene(
    scene_id, scene_gt, scene_gt_info, scene_gt_valid, scene_errs, correct_th, n_top
):
    """Matches the estimated poses to the ground-truth poses in one scene.

    :param scene_id: Scene ID.
    :param scene_gt: Dictionary mapping image ID's to lists of dictionaries with:
      - 'obj_id': Object ID of the ground-truth pose.
    :param scene_gt_valid: Dictionary mapping image ID's to lists of boolean
      values indicating which ground-truth poses should be considered.
    :param scene_errs: List of dictionaries with:
      - 'im_id': Image ID.
      - 'obj_id': Object ID.
      - 'est_id': ID of the pose estimate.
      - 'score': Confidence score of the pose estimate.
      - 'errors': Dictionary mapping ground-truth ID's to errors of the pose
          estimate w.r.t. the ground-truth poses.
    :param error_obj_threshs: Dictionary mapping object ID's to values of the
      threshold of correctness.
    :param n_top: Top N pose estimates (with the highest score) to be evaluated
      for each object class in each image.
    :return:
    """
    # Organize the errors by image ID and object ID (for faster query).
    scene_errs_org = {}
    for e in scene_errs:
        scene_errs_org.setdefault(e["im_id"], {}).setdefault(e["obj_id"], []).append(e)

    # Matching of poses in individual images.
    scene_matches = []
    for im_id, im_gts in scene_gt.items():
        im_matches = []

        for gt_id, gt in enumerate(im_gts):
            
            im_matches.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": gt["obj_id"],
                    "gt_id": gt_id,
                    "est_id": -1,
                    "score": -1,
                    "error": -1,
                    "error_norm": -1,
                    "valid": scene_gt_valid[im_id][gt_id],
                    "gt_visib_fract": scene_gt_info[im_id][gt_id]["visib_fract"], 
                }
            )

        # Treat estimates of each object separately.
        im_obj_ids = set([gt["obj_id"] for gt in im_gts])
        for obj_id in im_obj_ids:
            if (
                im_id in scene_errs_org.keys()
                and obj_id in scene_errs_org[im_id].keys()
            ):
                # Greedily match the estimated poses to the ground truth poses.
                errs_im_obj = scene_errs_org[im_id][obj_id]
                ms = match_poses(errs_im_obj, correct_th, n_top, scene_gt_valid[im_id])

                # Update info about the matched GT poses.
                for m in ms:
                    g = im_matches[m["gt_id"]]
                    g["est_id"] = m["est_id"]
                    g["score"] = m["score"]
                    g["error"] = m["error"]
                    g["error_norm"] = m["error_norm"]

        scene_matches += im_matches

    return scene_matches
