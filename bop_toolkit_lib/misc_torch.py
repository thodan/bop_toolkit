import torch


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: Bx3x3 ndarray with a rotation matrix.
    :param t: Bx3x1 ndarray with a translation vector.
    :return: Bxnx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    R_transposed = R.permute(0, 2, 1)  # Transpose R for batch multiplication
    t_transposed = t.permute(0, 2, 1)  # Bx3x1 to Bx1x3
    batch_t_transposed = t_transposed.repeat(1, pts.shape[0], 1)  # Bxnx3
    batch_pts = pts.unsqueeze(0).repeat(R.shape[0], 1, 1)  # Bxnx3
    batch_pts = torch.bmm(batch_pts, R_transposed) + batch_t_transposed
    return batch_pts  # Bxnx3


def project_pts(pts, K, R, t):
    """Projects 3D points.

    :param pts: nx3 tensor with the 3D points.
    :param K: Bx3x3 tensor with an intrinsic camera matrix.
    :param R: Bx3x3 tensor with a rotation matrix.
    :param t: Bx3x1 tensor with a translation vector.
    :return: Bxnx2 tensor with 2D image coordinates of the projections.
    """
    assert pts.shape[1] == 3
    P = torch.bmm(K, torch.cat((R, t), dim=2))
    pts_h = torch.cat(
        (pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)), dim=1
    )
    batch_pts_h = pts_h.unsqueeze(0).repeat(P.shape[0], 1, 1)  # Bxnx4
    batch_pts_im = torch.bmm(batch_pts_h, P.permute(0, 2, 1))
    batch_z = batch_pts_im[:, :, 2:].clone()
    batch_pts_im[:, :, :2] /= batch_z.repeat(1, 1, 2)
    return batch_pts_im[:, :, :2]  # Bxnx2
