import numpy as np
from bop_toolkit_lib.misc import project_pts_htt 
from hand_tracking_toolkit.camera import CameraModel



def mspd(R_est, t_est, R_gt, t_gt, cam: CameraModel, pts, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param cam: Hand Tracking Toolkit CameraModel object.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = project_pts_htt(pts, cam, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = project_pts_htt(pts, cam, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
    return min(es)
