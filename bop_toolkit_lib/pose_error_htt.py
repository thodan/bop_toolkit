import numpy as np
from bop_toolkit_lib.misc import transform_pts_Rt
from hand_tracking_toolkit.camera import model_by_name, CameraModel 


def create_camera_model(camera: dict):
    """
    Create a Hand Tracking Toolkit Camera model from a scene camera.
    """
    if "cam_K" in camera:        
        K = camera["cam_K"]            
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        width, height = 1,1
        model = "PinholePlane"
        coeffs = ()
    
    elif "cam_model" in camera:
        calib = camera["cam_model"]
        width = calib["image_width"]
        height = calib["image_height"]
        model = calib["projection_model_type"]

        if model == "CameraModelType.FISHEYE624" and len(calib["projection_params"]) == 15:
            # TODO: Aria data hack
            f, cx, cy = calib["projection_params"][:3]
            fx = fy = f
            coeffs = calib["projection_params"][3:]
        else:
            fx, fy, cx, cy = calib["projection_params"][:4]
            coeffs = calib["projection_params"][4:]

    else:
        raise ValueError("Scene camera data missing 'cam_K' or 'cam_model' fields")

    cls = model_by_name[model]
    return cls(
        width,
        height,
        (fx, fy),
        (cx, cy),
        coeffs
    )


def project_pts_htt(pts, cam: CameraModel, R, t):
    """Transform and projects points with Hand Tracking Toolbox CameraModel.

    :param pts: nx3 ndarray with the 3D points.
    :param cam: HTT CameraModel instance.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """

    pts_w = transform_pts_Rt(pts, R, t)
    pts_im = cam.eye_to_window(pts_w)

    return pts_im 


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
