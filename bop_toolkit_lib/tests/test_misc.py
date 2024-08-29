import torch
import numpy as np
import unittest
from bop_toolkit_lib import misc
from bop_toolkit_lib import misc_torch as misct
from bop_toolkit_lib import pose_error_htt
from bop_toolkit_lib import transform

from hand_tracking_toolkit.camera import PinholePlaneCameraModel


class TestMisc(unittest.TestCase):

    """
    Non regression tests for misc module.

    Checks that transform_pts_Rt and project_pts give similar
    results in sequential and batched implementations.
    """

    def setUp(self) -> None:
        self.Np = 100
        self.B = 10  # batch of rigid transformations (R,t)
        self.pts = torch.rand(self.Np,3)

    def test_transform_pts_Rt(self):
        """
        Sample poses at random and transform point cloud.
        """

        R_np = np.array([transform.random_rotation_matrix()[:3,:3] for _ in range(self.B)])
        t_np = np.random.rand(self.B,3,1)
        R_ts = torch.Tensor(R_np)
        t_ts = torch.Tensor(t_np)

        pts_np = np.zeros((self.B,self.Np,3))

        for i in range(self.B):
            pts_np[i] = misc.transform_pts_Rt(self.pts, R_np[i], t_np[i])
        pts_ts = misct.transform_pts_Rt(self.pts, R_ts, t_ts)

        self.assertTrue(np.allclose(pts_ts.numpy(), pts_np, atol=1e-6))

    def test_project_pts(self):

        """
        Compare 3 pinhole projection implementations:
        - misc.project_pts: numpy implementation
        - misc_torch.project_pts: torch implementation, enables batching accross rigid transformations
        - misc.project_pts_htt: Hand Tracking Toolkit implementation with CameraModel API

        For this test, camera rotation and translation are sampled
        a reference pose that looks toward the point cloud to avoid
        numerical issues for points st Zc <= 0.
        """
        R_ref = np.array([
            0,-1, 0,
            0, 0,-1,
            1, 0, 0
        ]).reshape((3,3))
        t_ref = np.array([0,0,2])
        sa, st = 0.1, 0.1
        R_np = np.array([R_ref@transform.euler_matrix(*(sa*np.random.random(3)))[:3,:3] 
                         for _ in range(self.B)])
        t_np = np.array([t_ref + st*np.random.random(3) for _ in range(self.B)])
        t_np = t_np.reshape((self.B,3,1))
        R_ts = torch.Tensor(R_np)
        t_ts = torch.Tensor(t_np)

        fx, fy, cx, cy = 600, 600, 320, 240
        K = np.array([
            fx, 0, cx,
            0, fy, cy,
            0,0,1
        ]).reshape((3,3))

        # misc pinhole projection
        proj_np = np.zeros((self.B,self.Np,2))
        for i in range(self.B):
            proj_np[i] = misc.project_pts(self.pts, K, R_np[i], t_np[i])

        # VS misc torch pinhole projection
        K_ts = torch.Tensor(K).unsqueeze(0).repeat((self.B,1,1))
        proj_ts = misct.project_pts(self.pts, K_ts, R_ts, t_ts)
        self.assertTrue(np.allclose(proj_ts.numpy(), proj_np, atol=1e-3))

        # VS Hand Tracking Dataset model
        camera = PinholePlaneCameraModel(width=fx, height=fy, f=(fx,fy), c=(cx,cy), distort_coeffs=())
        proj_htt = np.zeros((self.B,self.Np,2))
        for i in range(self.B):
            proj_htt[i] = pose_error_htt.project_pts_htt(self.pts, camera, R_np[i], t_np[i])
        self.assertTrue(np.allclose(proj_htt, proj_np, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
