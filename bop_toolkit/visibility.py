# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Estimation of the visible object surface from depth images."""

import numpy as np


def _estimate_visib_mask(d_test, d_model, delta):
  """Estimates a mask of the visible object surface.

  :param d_test: Distance image of a scene in which the visibility is estimated.
  :param d_model: Rendered distance image of the object model.
  :param delta: Tolerance used in the visibility test.
  :return: Visibility mask.
  """
  assert (d_test.shape == d_model.shape)
  mask_valid = np.logical_and(d_test > 0, d_model > 0)

  d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
  visib_mask = np.logical_and(d_diff <= delta, mask_valid)

  return visib_mask


def estimate_visib_mask_gt(d_test, d_gt, delta):
  """Estimates a mask of the visible object surface in the ground-truth pose.

  :param d_test: Distance image of a scene in which the visibility is estimated.
  :param d_gt: Rendered distance image of the object model in the GT pose.
  :param delta: Tolerance used in the visibility test.
  :return: Visibility mask.
  """
  visib_gt = _estimate_visib_mask(d_test, d_gt, delta)
  return visib_gt


def estimate_visib_mask_est(d_test, d_est, visib_gt, delta):
  """Estimates a mask of the visible object surface in the estimated pose.

  For an explanation of why the visibility mask is calculated differently for
  the estimated and the ground-truth pose, see equation (14) and related text in
  Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.

  :param d_test: Distance image of a scene in which the visibility is estimated.
  :param d_est: Rendered distance image of the object model in the est. pose.
  :param visib_gt: Visibility mask of the object model in the GT pose (from
    function estimate_visib_mask_gt).
  :param delta: Tolerance used in the visibility test.
  :return: Visibility mask.
  """
  visib_est = _estimate_visib_mask(d_test, d_est, delta)
  visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
  return visib_est
