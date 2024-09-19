# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculation of performance scores."""

import numpy as np
from collections import defaultdict

from bop_toolkit_lib import misc


def calc_ap(rec, pre, coco_interpolation=None):
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
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L507
    :return: Average Precision - the area under the monotonically decreasing
             version of the precision/recall curve given by rec and pre.
    """
    # Sorts the precision/recall points by increasing recall.
    i = np.argsort(rec)

    mrec = np.concatenate(([0], np.array(rec)[i], [1]))
    mpre = np.concatenate(([0], np.array(pre)[i], [0]))
    assert mrec.shape == mpre.shape
    for i in range(mpre.size - 3, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    if not coco_interpolation:
        i = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    else:
        # Interpolate precision at specified recall thresholds
        # https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/#H3Calc
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
    errs,
    n_top,
    visib_gt_min,
    ignore_object_visible_less_than_visib_gt_min,
    do_print=True,
    double_check_size=False,
):
    """Calculates performance scores for the 6D object detection task.

    References:
    Hodan et al., BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.
    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.

    :param scene_ids: ID's of considered scenes.
    :param obj_ids: ID's of considered objects.
    :param matches: Info about matching pose estimates to ground-truth poses
      (see pose_matching.py for details).
    :param scene_errs: List of dictionaries with:
      - 'im_id': Image ID.
      - 'obj_id': Object ID.
      - 'est_id': ID of the pose estimate.
      - 'score': Confidence score of the pose estimate.
      - 'errors': Dictionary mapping ground-truth ID's to errors of the pose
          estimate w.r.t. the ground-truth poses.
    :param n_top: Number of top pose estimates to consider per test target.
    :param visib_gt_min: Min visiblity for GT. Default: 0.1
    :param ignore_object_visible_less_than_visib_gt_min: Whether ignore objects visible less than visib_gt_min. Default: True
    :param do_print: Whether to print the scores to the standard output.
    :return: Dictionary with the evaluation scores.
    """
    # Count the number of visible object instances in each image.
    insts = {i: {j: defaultdict(lambda: 0) for j in scene_ids} for i in obj_ids}
    for m in matches:
        if m["valid"]:
            insts[m["obj_id"]][m["scene_id"]][m["im_id"]] += 1

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

    # For each object, sort the pose estimates by confidence score.
    # Then calculate the average precision for each object.
    scores_per_object = {}
    num_instances_per_object = {}
    total_num_false_ignore = 0
    for obj_id, obj_tar in obj_tars.items():
        obj_matches = [m for m in matches if m["obj_id"] == obj_id]

        # Sorting the predictions by confidences score and calculate TP, FP.
        sorted_obj_matches = sorted(obj_matches, key=lambda x: x["score"], reverse=True)

        # There are four types of matches:
        # True Positive: A correct detection with GT having visib_gt_fract >= visib_gt_min.
        true_positives = np.zeros(len(sorted_obj_matches), dtype=np.bool_)

        # False Positive: A detection that has no matching GT having visib_gt_fract >= visib_gt_min.
        false_positives = np.zeros(len(sorted_obj_matches), dtype=np.bool_)

        # False Negative: A GT having visib_gt_fract >= visib_gt_min, has no matching detection.
        false_negatives = np.zeros(len(sorted_obj_matches), dtype=np.bool_)

        # False Positive Ignore: A detection that has no matching GT having visib_gt_fract < visib_gt_min.
        false_ignore = np.zeros(len(sorted_obj_matches), dtype=np.bool_)

        for i, m in enumerate(sorted_obj_matches):
            # valid is object_id in target list
            # est_id != -1 is when there is a match, -1 otherwise

            # Case 1: Detection is a match.
            if m["est_id"] != -1: 
                if m["valid"]:
                    if m["gt_visib_fract"] >= visib_gt_min:
                        true_positives[i] = True
                    elif m["gt_visib_fract"] < visib_gt_min and ignore_object_visible_less_than_visib_gt_min:
                        false_ignore[i] = True
                    else:
                        print("Warning: not using ignore_object_visible_less_than_visib_gt_min")
                else:
                    # est_id != -1 and valid is always together.
                    print(f"Warning: not using valid {m}")
            # Case 2: Detection is not a match and it is a valid GT.
            elif m["valid"]:
                if m["gt_visib_fract"] >= visib_gt_min:
                    false_negatives[i] = True
                elif m["gt_visib_fract"] < visib_gt_min:
                    false_ignore[i] = True
            # Case 3: Detection is not a match and it is not a valid GT.
            else:
                false_positives[i] = True
    
        if double_check_size:
            # Double check the size of GT is correct
            num_gts = len([m for m in obj_matches if m["valid"]])
            assert np.sum(true_positives) + np.sum(false_negatives) + np.sum(false_ignore) == num_gts, f"TP={np.sum(true_positives)}, FN={np.sum(false_negatives)}, num_gts={num_gts}"

        # remove the false positives that are ignored
        keep_idx = np.invert(false_ignore)
        true_positives = true_positives[keep_idx]
        false_positives = false_positives[keep_idx]
        obj_tar = obj_tar - np.sum(false_ignore)

        cum_true_positives = np.cumsum(true_positives)
        cum_false_positives = np.cumsum(false_positives)

        # Recall, Precision.
        recall = cum_true_positives / int(obj_tar)
        precision = cum_true_positives / (cum_true_positives + cum_false_positives)
        precision[np.isnan(precision)] = 0

        ap = calc_ap(recall, precision, coco_interpolation=True)
        scores_per_object[obj_id] = ap
        num_instances_per_object[obj_id] = int(obj_tar)
        if do_print:
            misc.log("Object {:d} AP: {:.4f}".format(obj_id, ap))
            if np.sum(false_ignore) > 0:
                misc.log(
                    f"Number of false ignored: {np.sum(false_ignore)}"
                )
        total_num_false_ignore += np.sum(false_ignore)

    # Final scores.
    scores = {
        "gt_count": len(matches),
        "targets_count": int(tars),
        "num_estimates": len(errs),
        "scores": scores_per_object,
        "num_instances_per_object": num_instances_per_object,
    }
    if do_print:
        misc.log("")
        misc.log("Estimates count:    {:d}".format(scores["num_estimates"]))
        misc.log("GT count:           {:d}".format(scores["gt_count"]))
        misc.log("Target count:       {:d}".format(scores["targets_count"]))
        misc.log("Total number of false ignored: {:d}".format(total_num_false_ignore))
        misc.log("")

    return scores


if __name__ == "__main__":
    # AP test.
    tp = np.array([False, True, True, False, True, False])
    fp = np.logical_not(tp)
    tp_c = np.cumsum(tp).astype(np.float64)
    fp_c = np.cumsum(fp).astype(np.float64)
    rec = tp_c / tp.size
    pre = tp_c / (fp_c + tp_c)
    misc.log("Average Precision: " + str(calc_ap(rec, pre)))
