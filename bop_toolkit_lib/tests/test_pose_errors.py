import os
from pathlib import Path
import unittest
import numpy as np
import torch

from bop_toolkit_lib.config import datasets_path
from bop_toolkit_lib.dataset_params import get_split_params, get_model_params
from bop_toolkit_lib.inout import load_ply, load_json, load_scene_gt, load_scene_camera
from bop_toolkit_lib.misc import get_symmetry_transformations

from bop_toolkit_lib import transform
from bop_toolkit_lib import pose_error, pose_error_gpu, pose_error_htt 

from hand_tracking_toolkit.camera import PinholePlaneCameraModel


assert "BOP_PATH" in os.environ, "BOP_PATH environment variable is not defined"
DS_NAME = 'ycbv'
DS_SPLIT = 'test'


def generate_random_pose_est(R, t, B):
    R_np = np.array([R@(transform.random_rotation_matrix()[:3,:3]) for _ in range(B)])
    t_np = t + np.random.rand(B,3,1)
    R_ts = torch.Tensor(R_np)
    t_ts = torch.Tensor(t_np)
    return R_np, t_np, R_ts, t_ts


class TestPoseErrors(unittest.TestCase):

    def setUp(self) -> None:
        self.B = 10

        # tranformation batch size
        split_params = get_split_params(datasets_path, DS_NAME, DS_SPLIT)
        model_params = get_model_params(datasets_path, DS_NAME)
        models_info = load_json(model_params["models_info_path"], keys_to_int=True)

        # get first GT annotation of the first image of the first scene 
        self.im_size = split_params["im_size"]
        scene_id = split_params["scene_ids"][0]
        scene_dir = Path(split_params["split_path"]) / "{scene_id:06d}".format(scene_id=scene_id)
        scene_gt = load_scene_gt(split_params["scene_gt_tpath"].format(scene_id=scene_id))
        img_id = sorted(scene_gt.keys())[0]
        obj_gt = scene_gt[img_id][0] 
        obj_id = obj_gt["obj_id"]
        self.R_gt = obj_gt["cam_R_m2c"]
        self.t_gt = obj_gt["cam_t_m2c"]
        self.R_gt_ts = torch.Tensor(self.R_gt).float() 
        self.t_gt_ts = torch.Tensor(self.t_gt).float()
        self.R_gt_ts = self.R_gt_ts.unsqueeze(0).repeat(self.B,1,1)
        self.t_gt_ts = self.t_gt_ts.unsqueeze(0).repeat(self.B,1,1)

        # get object model
        self.model = load_ply(model_params["model_tpath"].format(obj_id=obj_id))
        self.syms = get_symmetry_transformations(models_info[obj_id], 0.01)
        self.syms_ts = [{
            "R": torch.from_numpy(sym["R"]).float(),
            "t": torch.from_numpy(sym["t"]).float(),
        } for sym in self.syms]
        self.pts = self.model["pts"]
        self.pts_ts = torch.tensor(self.pts).float()

        # get camera model
        scene_camera = load_scene_camera(scene_dir / "scene_camera.json")[scene_id]
        if "cam_K" in scene_camera:
            self.K = scene_camera["cam_K"]
            self.K_ts = torch.Tensor(self.K).unsqueeze(0).repeat(self.B, 1, 1).float()

            fx, fy = self.K[0,0], self.K[1,1]
            cx, cy = self.K[0,2], self.K[1,2]
            self.camera = PinholePlaneCameraModel(width=self.im_size[0], height=self.im_size[1], f=(fx,fy), c=(cx,cy), distort_coeffs=())
        else:
            raise NotImplementedError

    def test_mssd(self):
        """Compare MSSD implementations: CPU vs batched GPU """
        R_np, t_np, R_ts, t_ts = generate_random_pose_est(self.R_gt, self.t_gt, self.B)

        err_np = np.zeros(self.B)
        for i in range(self.B):
            err_np[i] = pose_error.mssd(R_np[i], t_np[i], self.R_gt, self.t_gt, self.pts, self.syms)
        err_ts = pose_error_gpu.mssd_by_batch(R_ts, t_ts, self.R_gt_ts, self.t_gt_ts, self.pts_ts, self.syms_ts)
        self.assertTrue(np.allclose(err_ts.numpy(), err_np, atol=1e-6))

    def test_mspd(self):
        """Compare Pinhole MSPD implementations: CPU vs batched GPU vs Hand Tracking Toolkit CameraModel """
        R_np, t_np, R_ts, t_ts = generate_random_pose_est(self.R_gt, self.t_gt, self.B)

        err_np = np.zeros(self.B)
        for i in range(self.B):
            err_np[i] = pose_error.mspd(R_np[i], t_np[i], self.R_gt, self.t_gt, self.K, self.pts, self.syms)
        err_ts = pose_error_gpu.mspd_by_batch(R_ts, t_ts, self.R_gt_ts, self.t_gt_ts, self.K_ts, self.pts_ts, self.syms_ts)
        self.assertTrue(np.allclose(err_ts.numpy(), err_np, atol=1e-6))

        # Hand Tracking Tookit CameraModel API
        err_htt = np.zeros(self.B)
        for i in range(self.B):
            err_htt[i] = pose_error_htt.mspd(R_np[i], t_np[i], self.R_gt, self.t_gt, self.camera, self.pts, self.syms)
        self.assertTrue(np.allclose(err_htt, err_np, atol=1e-4))
        

if __name__ == "__main__":
    unittest.main()
