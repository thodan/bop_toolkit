# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Estimation of the visible object surface from depth images."""

import numpy as np


def _estimate_visib_mask(d_test, d_model, delta, visib_mode="bop19"):
    """Estimates a mask of the visible object surface.

    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: Visibility mode:
    1) 'bop18' - Object is considered NOT VISIBLE at pixels with missing depth.
    2) 'bop19' - Object is considered VISIBLE at pixels with missing depth. This
         allows to use the VSD pose error function also on shiny objects, which
         are typically not captured well by the depth sensors. A possible problem
         with this mode is that some invisible parts can be considered visible.
         However, the shadows of missing depth measurements, where this problem is
         expected to appear and which are often present at depth discontinuities,
         are typically relatively narrow and therefore this problem is less
         significant.
    :return: Visibility mask.
    """
    assert d_test.shape == d_model.shape

    if visib_mode == "bop18":
        mask_valid = np.logical_and(d_test > 0, d_model > 0)
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    elif visib_mode == "bop19":
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(
            np.logical_or(d_diff <= delta, d_test == 0), d_model > 0
        )

    else:
        raise ValueError("Unknown visibility mode.")

    return visib_mask


def estimate_visib_mask_gt(d_test, d_gt, delta, visib_mode="bop19"):
    """Estimates a mask of the visible object surface in the ground-truth pose.

    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_gt: Rendered distance image of the object model in the GT pose.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_gt = _estimate_visib_mask(d_test, d_gt, delta, visib_mode)
    return visib_gt


def estimate_visib_mask_est(d_test, d_est, visib_gt, delta, visib_mode="bop19"):
    """Estimates a mask of the visible object surface in the estimated pose.

    For an explanation of why the visibility mask is calculated differently for
    the estimated and the ground-truth pose, see equation (14) and related text in
    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.

    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_est: Rendered distance image of the object model in the est. pose.
    :param visib_gt: Visibility mask of the object model in the GT pose (from
      function estimate_visib_mask_gt).
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_est = _estimate_visib_mask(d_test, d_est, delta, visib_mode)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
    return visib_est
