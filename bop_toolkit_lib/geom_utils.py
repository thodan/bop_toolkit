import numpy as np


def backproj_depth(depth, intrinsics, mask=None):
    intrinsics_inv = np.linalg.inv(intrinsics)
    val_depth = depth > 0
    if mask is None:
        val_mask = val_depth
    else:
        val_mask = np.logical_and(mask, val_depth)
    idxs = np.where(val_mask)
    grid = np.array([idxs[1], idxs[0]])
    ones = np.ones([1, grid.shape[1]])
    uv_grid = np.concatenate((grid, ones), axis=0)
    xyz = intrinsics_inv @ uv_grid
    xyz = np.transpose(xyz).squeeze()
    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    return pts, idxs
