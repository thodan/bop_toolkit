# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculation of performance scores."""

import numpy as np
from collections import defaultdict

from bop_toolkit_lib import misc


def calc_ap(rec, pre):
    """Calculates Average Precision (AP).

    Calculated in the PASCAL VOC challenge from 2010 onwards [1]:
    1) Compute a version of the measured precision/recall curve with precision
       monotonically decreasing, by setting the precision for recall r to the
       maximum precision obtained for any recall r' >= r.
    2) Compute the AP as the area under this curve by numerical integration.
       No approximation is involved since the curve is piecewise constant.

    NOTE: The used AP formula is different from the one in [2] where the
    formula from VLFeat [3] was presented - although it was mistakenly
    introduced as a formula used in PASCAL.

    References:
    [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00044000000000000000
    [2] Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016
    [3] http://www.vlfeat.org/matlab/vl_pr.html

    :param rec: A list (or 1D ndarray) of recall rates.
    :param pre: A list (or 1D ndarray) of precision rates.
    :param coco_interpolation: If True, do interpolation at these recall thresholds as done in COCO
    :return: Average Precision - the area under the monotonically decreasing
             version of the precision/recall curve given by rec and pre.
    """
    # Sorts the precision/recall points by increasing recall.
    i = np.argsort(rec)
    mrec = np.concatenate(([0], np.array(rec)[i], [1]))
    mpre = np.concatenate(([0], np.array(pre)[i], [0]))
    assert mrec.shape == mpre.shape

    # Follow COCO API and interpolate at these recall thresholds.
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L507
    rec_thresholds = np.linspace(0.0, 1.00, 101, endpoint=True)
    interpolated_precisions = []
    for rt in rec_thresholds:
        if np.any(mrec >= rt):
            interpolated_precisions.append(np.max(mpre[mrec >= rt]))
        else:
            interpolated_precisions.append(0.0)
    ap = np.mean(interpolated_precisions)
    return ap


def calc_recall(tp_count, targets_count):
    """Calculates recall.

    :param tp_count: Number of true positives.
    :param targets_count: Number of targets.
    :return: The recall rate.
    """
    if targets_count == 0:
        return 0.0
    else:
        return tp_count / float(targets_count)


def calc_localization_scores(scene_ids, obj_ids, matches, n_top, do_print=True):
    """Calculates performance scores for the 6D object localization task.

    References:
    Hodan et al., BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.
    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.

    :param scene_ids: ID's of considered scenes.
    :param obj_ids: ID's of considered objects.
    :param matches: Info about matching pose estimates to ground-truth poses
      (see pose_matching.py for details).
    :param n_top: Number of top pose estimates to consider per test target.
    :param do_print: Whether to print the scores to the standard output.
    :return: Dictionary with the evaluation scores.
    """
    # Count the number of visible object instances in each image.
    insts = {i: {j: defaultdict(lambda: 0) for j in scene_ids} for i in obj_ids}
    for m in matches:
        if m["valid"]:
            insts[m["obj_id"]][m["scene_id"]][m["im_id"]] += 1

    # Count the number of targets = object instances to be found.
    # For SiSo, there is either zero or one target in each image - there is just
    # one even if there are more instances of the object of interest.
    tars = 0  # Total number of targets.
    obj_tars = {i: 0 for i in obj_ids}  # Targets per object.
    scene_tars = {i: 0 for i in scene_ids}  # Targets per scene.
    for obj_id, obj_insts in insts.items():
        for scene_id, scene_insts in obj_insts.items():
            # Count the number of targets for the current object in the current scene.
            if n_top > 0:
                count = sum(np.minimum(n_top, list(scene_insts.values())))
            else:
                count = sum(list(scene_insts.values()))

            tars += count
            obj_tars[obj_id] += count
            scene_tars[scene_id] += count

    # Count the number of true positives.
    tps = 0  # Total number of true positives.
    obj_tps = {i: 0 for i in obj_ids}  # True positives per object.
    scene_tps = {i: 0 for i in scene_ids}  # True positives per scene.
    for m in matches:
        if m["valid"] and m["est_id"] != -1:
            tps += 1
            obj_tps[m["obj_id"]] += 1
            scene_tps[m["scene_id"]] += 1

    # Total recall.
    recall = calc_recall(tps, tars)

    # Recall per object.
    obj_recalls = {}
    for i in obj_ids:
        obj_recalls[i] = calc_recall(obj_tps[i], obj_tars[i])
    mean_obj_recall = float(np.mean(list(obj_recalls.values())).squeeze())

    # Recall per scene.
    scene_recalls = {}
    for i in scene_ids:
        scene_recalls[i] = float(calc_recall(scene_tps[i], scene_tars[i]))
    mean_scene_recall = float(np.mean(list(scene_recalls.values())).squeeze())

    # Final scores.
    scores = {
        "recall": float(recall),
        "obj_recalls": obj_recalls,
        "mean_obj_recall": float(mean_obj_recall),
        "scene_recalls": scene_recalls,
        "mean_scene_recall": float(mean_scene_recall),
        "gt_count": len(matches),
        "targets_count": int(tars),
        "tp_count": int(tps),
    }

    if do_print:
        obj_recalls_str = ", ".join(
            ["{}: {:.3f}".format(i, s) for i, s in scores["obj_recalls"].items()]
        )

        scene_recalls_str = ", ".join(
            ["{}: {:.3f}".format(i, s) for i, s in scores["scene_recalls"].items()]
        )

        misc.log("")
        misc.log("GT count:           {:d}".format(scores["gt_count"]))
        misc.log("Target count:       {:d}".format(scores["targets_count"]))
        misc.log("TP count:           {:d}".format(scores["tp_count"]))
        misc.log("Recall:             {:.4f}".format(scores["recall"]))
        misc.log("Mean object recall: {:.4f}".format(scores["mean_obj_recall"]))
        misc.log("Mean scene recall:  {:.4f}".format(scores["mean_scene_recall"]))
        misc.log("Object recalls:\n{}".format(obj_recalls_str))
        misc.log("Scene recalls:\n{}".format(scene_recalls_str))
        misc.log("")

    return scores


def calc_pose_detection_scores(
    scene_ids,
    obj_ids,
    matches,
    ests_info,
    visib_gt_min,
    do_print=True,
):
    """Calculates accuracy scores for the 6D object detection task.
    :param scene_ids: ID's of considered scenes.
    :param obj_ids: ID's of considered objects.
    :param matches: List of per-GT dictionary with info about GT-to-detection matching.
      (see pose_matching.py for details).
    :param ests_info: List of per-detection dictionary with the following items:
      - 'im_id': Image ID.
      - 'obj_id': Object ID.
      - 'est_id': ID of the pose estimate.
      - 'score': Confidence score of the pose estimate.
      - 'errors': Dictionary mapping ground-truth ID's to errors of the pose
          estimate w.r.t. the ground-truth poses.
    :param visib_gt_min: GT annotations visible from less than this threshold are ignored.
        from less than visib_gt_min.
    :param do_print: Whether to print the scores to the standard output.
    :return: Dictionary with the per-object AP scores.
    """

    scores_per_object = {}
    num_targets_per_object = {}

    for obj_id in obj_ids:

        # Get mapping from GTs (keep only those that are visible enough) to matched detections.
        obj_gt_to_est_matching = []
        est_signatures_matched = []
        est_signatures_to_ignore = []
        for match in matches:
            if match["obj_id"] == obj_id:
                
                # Unique signature of the estimate.
                est_signature = (match["scene_id"], match["im_id"], match["obj_id"], match["est_id"])

                if match["gt_visib_fract"] >= visib_gt_min:
                    # Consider only GTs which are sufficiently visible.
                    obj_gt_to_est_matching.append(match)

                    if match["est_id"] != -1:
                        est_signatures_matched.append(est_signature)
                else:
                    # Detections matched to invisible GTs will be ignored.
                    est_signatures_to_ignore.append(est_signature)

        # Number of valid GT annotations for the current object ID.
        num_obj_gts = len(obj_gt_to_est_matching)
        num_targets_per_object[obj_id] = num_obj_gts

        # Detections of the current object which are not associated with any ignored GT.
        obj_ests = []
        for est in ests_info:
            est_signature = (est["scene_id"], est["im_id"], est["obj_id"], est["est_id"])
            if est["obj_id"] == obj_id and est_signature not in est_signatures_to_ignore:
                est["matched"] = est_signature in est_signatures_matched
                obj_ests.append(est)

        # Number of valid detections for the current object.
        num_obj_ests = len(obj_ests)

        # Sort detections by confidence scores (necessary for calculating precion-recall curve).
        obj_ests_sorted = sorted(obj_ests, key=lambda x: x["score"], reverse=True)

        # Binary indicators of detection types:
        true_positive_mask = np.zeros(num_obj_ests, dtype=np.bool_)
        false_positive_mask = np.zeros(num_obj_ests, dtype=np.bool_)

        # We iterate over sorted estimates and set the masks.
        for i, est in enumerate(obj_ests_sorted):
            if est["matched"]:
                true_positive_mask[i] = True
            else:
                false_positive_mask[i] = True

        # Cumulative sums (this can be done as we sorted the matches at the beginning).
        true_positive_cumsum = np.cumsum(true_positive_mask)
        false_positive_cumsum = np.cumsum(false_positive_mask)

        # Calculate recall values.
        recalls = true_positive_cumsum / float(num_obj_gts)

        # Calculate precision values.
        precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum)
        precisions[np.isinf(precisions)] = 0.0
        precisions[np.isnan(precisions)] = 0.0

        # Calculate AP (Average Precision).
        obj_ap = calc_ap(recalls, precisions)
        scores_per_object[obj_id] = obj_ap

        if do_print:
            misc.log(f"Object {obj_id:d} AP: {obj_ap:.4f}")

    return {
        "scores": scores_per_object,
        "num_targets_per_object": num_targets_per_object,
    }


if __name__ == "__main__":
    # AP test.
    tp = np.array([False, True, True, False, True, False])
    fp = np.logical_not(tp)
    tp_c = np.cumsum(tp).astype(np.float64)
    fp_c = np.cumsum(fp).astype(np.float64)
    rec = tp_c / tp.size
    pre = tp_c / (fp_c + tp_c)
    misc.log("Average Precision: " + str(calc_ap(rec, pre)))
