import torch
import numpy as np
import unittest
from bop_toolkit_lib import misc
from bop_toolkit_lib import misc_torch as misct
from bop_toolkit_lib import transform


# project_pts
# transform_pts_Rt



class TestMisc(unittest.TestCase):

    def setUp(self) -> None:
        self.Np = 100
        self.B = 10

        self.pts = torch.rand(self.Np,3)

    def test_transform_pts_Rt(self):

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
        For this test, we cannot sample poses completely at random:
        Points close to the camera plane Zc~0 will create issues
        Instead, sample camera rotation and translation around
        a reference pose that looks toward the point cloud
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
        K_ts = torch.Tensor(K).unsqueeze(0).repeat((self.B,1,1))

        proj_np = np.zeros((self.B,self.Np,2))

        for i in range(self.B):
            proj_np[i] = misc.project_pts(self.pts, K, R_np[i], t_np[i])
        proj_ts = misct.project_pts(self.pts, K_ts, R_ts, t_ts)

        # eh = np.allclose(proj_ts.numpy(), proj_np, atol=1e-3)
        # # if not eh:
        # #     print(eh)
        # #     breakpoint()

        self.assertTrue(np.allclose(proj_ts.numpy(), proj_np, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
