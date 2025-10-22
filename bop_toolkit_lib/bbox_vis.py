import copy

import cv2
import numpy as np
from bop_toolkit_lib import misc
from bop_toolkit_lib.common_utils import adjust_img_for_plt, cast_to_numpy
from bop_toolkit_lib.vis_utils import draw_text_in_ul


def draw_pose_on_img(
    rgb,
    K,
    pose_pred,
    mesh_bbox=None,
    bbox_color=(255, 255, 0),
    bbox_color_gt=(0, 255, 0),
    axes_scale=50.0,
    pose_gt=None,
    final_frame=None,
    extra_text=None,
):
    if mesh_bbox is not None:
        mesh_bbox = np.array(mesh_bbox) if isinstance(mesh_bbox, list) else mesh_bbox
    if (
        pose_pred.shape[0] == 1
        and mesh_bbox is not None
        and mesh_bbox.ndim < pose_pred.ndim
    ):
        mesh_bbox = mesh_bbox[None]
    if len(pose_pred.shape) == 3:
        final_frame = None
        if mesh_bbox is not None:
            mesh_bbox = (
                np.array(mesh_bbox) if isinstance(mesh_bbox, list) else mesh_bbox
            )
            assert len(mesh_bbox.shape) == 3, f"{mesh_bbox.shape=}"
        for idx in range(len(pose_pred)):
            final_frame = draw_pose_on_img(
                rgb,
                K,
                pose_pred[idx],
                mesh_bbox=None if mesh_bbox is None else mesh_bbox[idx],
                bbox_color=bbox_color,
                bbox_color_gt=bbox_color_gt,
                axes_scale=axes_scale,
                pose_gt=None if pose_gt is None else pose_gt[idx],
                final_frame=final_frame,
                extra_text=extra_text,
            )
        return final_frame

    rgb = adjust_img_for_plt(rgb) if final_frame is None else final_frame
    K = cast_to_numpy(K)
    pose_pred = cast_to_numpy(pose_pred)
    if np.all(pose_pred == np.eye(4)):
        print("Estimated pose is identity. Not drawing")
        return rgb
    final_frame = draw_xyz_axis(
        rgb, scale=axes_scale, K=K, rt=pose_pred, is_input_rgb=True
    )
    if mesh_bbox is not None:
        final_frame = draw_posed_3d_box(
            final_frame, rt=pose_pred, K=K, bbox=mesh_bbox, line_color=bbox_color
        )
        if pose_gt is not None:
            pose_gt = cast_to_numpy(pose_gt)
            final_frame = draw_posed_3d_box(
                final_frame, rt=pose_gt, K=K, bbox=mesh_bbox, line_color=bbox_color_gt
            )
    if extra_text is not None:
        if isinstance(extra_text, list):
            extra_text = "\n".join(extra_text)
        final_frame = draw_text_in_ul(final_frame, extra_text, size=1, thickness=3)
    return final_frame


def draw_xyz_axis(
    rgb,
    rt,
    K,
    scale=10.0,
    thickness=4,
    transparency=0,
    is_input_rgb=False,
    do_add_text=False,
):
    """
    based on: https://github.com/NVlabs/FoundationPose/blob/main/Utils.py
    """
    if is_input_rgb:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    R = rt[:3, :3]
    t = rt[:3, 3].reshape(-1, 1)
    xx = np.array([[1, 0, 0.0]])
    yy = np.array([[0, 1, 0.0]])
    zz = np.array([[0, 0, 1.0]])
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = misc.project_pts(np.array([[0, 0, 1.0]]), K, R, t).squeeze().astype(int)
    xx = misc.project_pts(xx, K, R, t).squeeze().astype(int)
    yy = misc.project_pts(yy, K, R, t).squeeze().astype(int)
    zz = misc.project_pts(zz, K, R, t).squeeze().astype(int)

    line_type = cv2.LINE_AA
    color_x = (0, 0, 255)
    color_y = (255, 255, 0)
    color_z = (255, 0, 0)
    arrow_len = 0

    tmp = rgb.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=color_x,
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=color_y,
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=color_z,
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp


def draw_posed_3d_box(img, rt, K, bbox, line_color=(0, 255, 0), linewidth=2):
    """
    based on: https://github.com/NVlabs/FoundationPose/blob/main/Utils.py
    """
    bbox = cast_to_numpy(bbox)
    if bbox.ndim == 3:
        bbox = bbox.squeeze(0)
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        uv = (
            misc.project_pts(pts, K, rt[:3, :3], t=rt[:3, 3].reshape(-1, 1))
            .squeeze()
            .astype(int)
        )
        img = cv2.line(
            img,
            uv[0].tolist(),
            uv[1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img
