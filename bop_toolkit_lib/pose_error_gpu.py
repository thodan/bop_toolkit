# Author: Van Nguyen Nguyen (vanngn.nguyen@gmail.com)
# Imagine team, Ecole des Ponts ParisTech

import numpy as np
import torch
import gc
from bop_toolkit_lib import misc_torch as misc

MAX_BATCH_SIZE = 400  # (1.0 GB in average, less than 2 GB for all BOP objects)


class BatchedData:
    # taken from https://github.com/nv-nguyen/gigapose/blob/f81a5413a912a0eae13c59b276ec4b41d4eca094/src/utils/batch.py
    """
    A structure for storing data in batched format to handle very large batch size.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        if isinstance(self.data, np.ndarray):
            return np.ceil(len(self.data) / self.batch_size).astype(int)
        elif isinstance(self.data, torch.Tensor):
            length = self.data.shape[0]
            return np.ceil(length / self.batch_size).astype(int)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


@torch.no_grad()
def mssd_by_batch(R_est, t_est, R_gt, t_gt, pts, syms, max_batch_size=MAX_BATCH_SIZE):
    """
    mssd with max_batch_size for R_est, t_est, R_gt, t_gt.
    This allows to stabilize the memory usage (1GB for batch_size=200).
    """
    batch_R_est = BatchedData(max_batch_size, R_est)
    batch_t_est = BatchedData(max_batch_size, t_est)
    batch_R_gt = BatchedData(max_batch_size, R_gt)
    batch_t_gt = BatchedData(max_batch_size, t_gt)
    output = BatchedData(batch_size=max_batch_size)
    for i in range(len(batch_R_est)):
        output_ = mssd(
            batch_R_est[i],
            batch_t_est[i],
            batch_R_gt[i],
            batch_t_gt[i],
            pts,
            syms,
        )
        output.cat(output_)
    return output.data


@torch.no_grad()
def mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: Bx3x3 ndarray with the estimated rotation matrix.
    :param t_est: Bx3x1 ndarray with the estimated translation vector.
    :param R_gt: Bx3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: Bx3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    es = []
    for sym in syms:
        batch_sym_R = sym["R"].unsqueeze(0).repeat(R_gt.shape[0], 1, 1)
        batch_sym_t = sym["t"].unsqueeze(0).repeat(t_gt.shape[0], 1, 1)
        R_gt_sym = torch.bmm(R_gt, batch_sym_R)
        t_gt_sym = torch.bmm(R_gt, batch_sym_t) + t_gt

        pts_gt_sym = misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        err = torch.norm(pts_est - pts_gt_sym, dim=2)
        max_err = err.max(dim=1).values
        es.append(max_err)
    es = torch.stack(es, dim=1).min(dim=1).values
    gc.collect()
    torch.cuda.empty_cache()
    return es


@torch.no_grad()
def mspd_by_batch(
    R_est, t_est, R_gt, t_gt, K, pts, syms, max_batch_size=MAX_BATCH_SIZE
):
    """
    mspd with max_batch_size for R_est, t_est, R_gt, t_gt.
    This allows to stabilize the memory usage (1GB for batch_size=200).
    """
    batch_R_est = BatchedData(max_batch_size, R_est)
    batch_t_est = BatchedData(max_batch_size, t_est)
    batch_R_gt = BatchedData(max_batch_size, R_gt)
    batch_t_gt = BatchedData(max_batch_size, t_gt)
    batch_K = BatchedData(max_batch_size, K)
    output = BatchedData(batch_size=max_batch_size)
    for i in range(len(batch_R_est)):
        output_ = mspd(
            batch_R_est[i],
            batch_t_est[i],
            batch_R_gt[i],
            batch_t_gt[i],
            batch_K[i],
            pts,
            syms,
        )
        output.cat(output_)
    return output.data


@torch.no_grad()
def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: Bx3x3 ndarray with the estimated rotation matrix.
    :param t_est: Bx3x1 ndarray with the estimated translation vector.
    :param R_gt: Bx3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: Bx3x1 ndarray with the ground-truth translation vector.
    :param K: Bx3x3 ndarray with the intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts, K, R_est, t_est)
    es = []
    for sym in syms:
        batch_sym_R = sym["R"].unsqueeze(0).repeat(R_gt.shape[0], 1, 1)
        batch_sym_t = sym["t"].unsqueeze(0).repeat(t_gt.shape[0], 1, 1)
        R_gt_sym = torch.bmm(R_gt, batch_sym_R)
        t_gt_sym = torch.bmm(R_gt, batch_sym_t) + t_gt

        proj_gt_sym = misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        err = torch.norm(proj_est - proj_gt_sym, dim=2)
        max_err = err.max(dim=1).values
        es.append(max_err)
    es = torch.stack(es, dim=1).min(dim=1).values
    gc.collect()
    torch.cuda.empty_cache()
    return es
