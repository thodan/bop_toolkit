# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Implementation of the pose error functions described in:
Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation", ECCV'18
Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW'16
"""

import math
import numpy as np
from scipy import spatial

from bop_toolkit import misc
from bop_toolkit import visibility


def vsd(R_est, t_est, R_gt, t_gt, depth_test, K, delta, tau, renderer, obj_id,
        cost_type='step'):
  """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param depth_test: hxw ndarray with the test depth image.
  :param K: 3x3 ndarray with a camera matrix.
  :param delta: Tolerance used for estimation of the visibility masks.
  :param tau: Misalignment tolerance.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param obj_id: Object identifier.
  :param cost_type: Type of the pixel-wise matching cost:
      'tlinear' - Used in the original definition of VSD in:
          Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
      'step' - Used for SIXD Challenge 2017 onwards.
  :return: The calculated error.
  """
  # Render depth images of the model in the estimated and the ground-truth pose.
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)
  depth_est = renderer.get_depth_image(obj_id)
  renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)
  depth_gt = renderer.get_depth_image(obj_id)

  # Convert depth images to distance images.
  dist_test = misc.depth_im_to_dist_im(depth_test, K)
  dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
  dist_est = misc.depth_im_to_dist_im(depth_est, K)

  # Visibility mask of the model in the ground-truth pose.
  visib_gt = visibility.estimate_visib_mask_gt(
    dist_test, dist_gt, delta)

  # Visibility mask of the model in the estimated pose.
  visib_est = visibility.estimate_visib_mask_est(
    dist_test, dist_est, visib_gt, delta)

  # Intersection and union of the visibility masks.
  visib_inter = np.logical_and(visib_gt, visib_est)
  visib_union = np.logical_or(visib_gt, visib_est)

  # Pixel-wise matching cost.
  costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
  if cost_type == 'step':
    costs = costs >= tau
  elif cost_type == 'tlinear':  # Truncated linear function.
    costs *= (1.0 / tau)
    costs[costs > 1.0] = 1.0
  else:
    raise ValueError('Unknown pixel matching cost.')

  # Visible Surface Discrepancy.
  visib_union_count = visib_union.sum()
  visib_comp_count = visib_union_count - visib_inter.sum()
  if visib_union_count > 0:
    e = (costs.sum() + visib_comp_count) / float(visib_union_count)
  else:
    e = 1.0
  return e


def add(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with no indistinguishable
  views - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
  pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
  e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
  return e


def adi(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
  pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)

  # Calculate distances to the nearest neighbors from vertices in the
  # ground-truth pose to vertices in the estimated pose.
  nn_index = spatial.cKDTree(pts_est)
  nn_dists, _ = nn_index.query(pts_gt, k=1)

  e = nn_dists.mean()
  return e


def re(R_est, R_gt):
  """Rotational Error.

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :return: The calculated error.
  """
  assert (R_est.shape == R_gt.shape == (3, 3))
  error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

  # Avoid invalid values due to numerical errors.
  error_cos = min(1.0, max(-1.0, error_cos))

  error = math.acos(error_cos)
  error = 180.0 * error / np.pi  # Convert [rad] to [deg].
  return error


def te(t_est, t_gt):
  """Translational Error.

  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :return: The calculated error.
  """
  assert (t_est.size == t_gt.size == 3)
  error = np.linalg.norm(t_gt - t_est)
  return error


def proj(R_est, t_est, R_gt, t_gt, K, pts):
  """Average distance of projections of object model vertices [px]
  - by Brachmann et al. (CVPR'16).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param K: 3x3 ndarray with a camera matrix.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  proj_est = misc.project_pts(pts, K, R_est, t_est)
  proj_gt = misc.project_pts(pts, K, R_gt, t_gt)
  e = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
  return e


def cou_mask(mask_est, mask_gt):
  """Complement over Union of 2D binary masks.

  :param mask_est: hxw ndarray with the estimated mask.
  :param mask_gt: hxw ndarray with the ground-truth mask.
  :return: The calculated error.
  """
  mask_est_bool = mask_est.astype(np.bool)
  mask_gt_bool = mask_gt.astype(np.bool)

  inter = np.logical_and(mask_gt_bool, mask_est_bool)
  union = np.logical_or(mask_gt_bool, mask_est_bool)

  union_count = float(union.sum())
  if union_count > 0:
    e = 1.0 - inter.sum() / union_count
  else:
    e = 1.0
  return e


def cou_mask_proj(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
  """Complement over Union of projected 2D masks.

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param K: 3x3 ndarray with a camera matrix.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param obj_id: Object identifier.
  :return: The calculated error.
  """
  # Render depth images of the model at the estimated and the ground-truth pose.
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)
  depth_est = renderer.get_depth_image(obj_id)
  renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)
  depth_gt = renderer.get_depth_image(obj_id)

  # Masks of the rendered model and their intersection and union.
  mask_est = depth_est > 0
  mask_gt = depth_gt > 0
  inter = np.logical_and(mask_gt, mask_est)
  union = np.logical_or(mask_gt, mask_est)

  union_count = float(union.sum())
  if union_count > 0:
    e = 1.0 - inter.sum() / union_count
  else:
    e = 1.0
  return e


def cou_bb(bb_est, bb_gt):
  """Complement over Union of 2D bounding boxes.

  :param bb_est: The estimated bounding box (x1, y1, w1, h1).
  :param bb_gt: The ground-truth bounding box (x2, y2, w2, h2).
  :return: The calculated error.
  """
  e = 1.0 - misc.iou(bb_est, bb_gt)
  return e


def cou_bb_proj(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
  """Complement over Union of projected 2D bounding boxes.

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param K: 3x3 ndarray with a camera matrix.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param obj_id: Object identifier.
  :return: The calculated error.
  """
  # Render depth images of the model at the estimated and the ground-truth pose.
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)
  depth_est = renderer.get_depth_image(obj_id)
  renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)
  depth_gt = renderer.get_depth_image(obj_id)

  # Masks of the rendered model and their intersection and union
  mask_est = depth_est > 0
  mask_gt = depth_gt > 0

  ys_est, xs_est = mask_est.nonzero()
  bb_est = misc.calc_2d_bbox(xs_est, ys_est, im_size=None, clip=False)

  ys_gt, xs_gt = mask_gt.nonzero()
  bb_gt = misc.calc_2d_bbox(xs_gt, ys_gt, im_size=None, clip=False)

  e = 1.0 - misc.iou(bb_est, bb_gt)
  return e
