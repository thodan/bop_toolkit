import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bop_toolkit_lib import misc
from bop_toolkit_lib.common_utils import adjust_img_for_plt, cast_to_numpy
from bop_toolkit_lib.geom_utils import backproj_depth
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_depth_diff_img(gt_depth, est_pose, gt_pose, cam, syms=None):
    hw = gt_depth.shape[-2:]
    gt_depth_xyz, _ = backproj_depth(gt_depth, cam)
    dists = compute_per_point_dists(
        gt_depth_xyz,
        get_pose_mat_from_dict(gt_pose),
        get_pose_mat_from_dict(est_pose),
        syms=syms,
    )
    depth_diff_img = np.zeros(hw)
    gt_depth_uv = np.where(gt_depth > 0)
    depth_diff_img[gt_depth_uv] = dists
    return depth_diff_img


def combine_depth_diffs(masks, diffs, use_clip=False, clip_val=None):
    combined = np.zeros_like(diffs[0], dtype=np.float32)
    imgs = []
    for i, mask in enumerate(masks):
        combined[mask] = diffs[i][mask]
        imgs.append(combined)
    if use_clip:
        clip_val = np.percentile(combined, 99.7) if clip_val is None else clip_val
        combined = np.clip(combined, 0, clip_val)
    return {
        "combined": combined,
        "imgs": imgs,
    }


def draw_pose_contour(
    cv_img,
    rendered_depth,
    contour_color=(255, 0, 0),
    thickness=3,
    mask_visib=None,
    use_depth=False,
):
    # based on: https://github.com/megapose6d/megapose6d/blob/master/src/megapose/visualization/utils.py

    mask = ((rendered_depth > 0).astype(np.uint8)) * 255

    if mask_visib is not None:
        mask = mask * mask_visib

    # clean up small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, k, iterations=1)

    if use_depth:
        depth2 = rendered_depth.copy()
        depth2[mask == 0] = 0
        assert depth2.max() > 1e1, f"Ensure the depth has mm-scale, {depth2.max=}"
        depth2 = (depth2 * 1e-3 * 255).clip(0, 255).astype(np.uint8)
        mask_depth = cv2.Canny(depth2, threshold1=30, threshold2=100)
        edge = cv2.subtract(mask_depth, eroded)
    else:
        edge = cv2.subtract(mask, eroded)
        mask = cv2.Canny(mask, threshold1=30, threshold2=100)

    # get a thicker outline
    if thickness > 1:
        k_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edge = cv2.dilate(edge, k_thick, iterations=1)

    overlay = cv_img.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay[edge > 0] = contour_color

    return overlay


def get_depth_map_and_obj_masks_from_renderings(render_out_per_obj):
    depth_map = np.zeros_like(list(render_out_per_obj.values())[0]["depth"])
    for idx, obj_res in render_out_per_obj.items():
        depth_obj = obj_res["depth"]
        is_fg = (depth_map == 0) | (depth_obj < depth_map)
        exists = depth_obj > 0
        mask_obj = exists & is_fg
        depth_map[mask_obj] = depth_obj[mask_obj]

    # get masks based on the complete depth map
    mask_objs = []
    for i, (idx, obj_res) in enumerate(render_out_per_obj.items()):
        depth_obj = obj_res["depth"]
        is_fg = (depth_map == 0) | (depth_obj <= depth_map)
        exists = depth_obj > 0
        mask_obj = exists & is_fg
        mask_objs.append(mask_obj)

    return {
        "depth_map": depth_map,
        "mask_objs": mask_objs,
    }


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


def calc_mask_visib_percent(mask_visib, valid_mask):
    total_pixels = np.sum(valid_mask)
    visible_pixels = np.sum(mask_visib)
    percent_visible = (visible_pixels / total_pixels) * 100
    return percent_visible


def draw_text_in_ul(
    rgb, extra_text, size=1, thickness=3, start_pos=(10, 30), color=(255, 0, 0)
):
    rgb = cv2.putText(
        copy.deepcopy(rgb),
        extra_text,
        start_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return rgb


def merge_masks(masks):
    return np.any(masks, axis=0)


def get_pose_mat_from_dict(pose):
    return get_pose_mat_from_rt(pose["R"], pose["t"])


def get_pose_mat_from_rt(rot, t):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rot
    pose_mat[:3, 3] = t.squeeze()
    return pose_mat


def compute_per_point_dists(pts_gt_cam, pose_gt, pose_est, syms=None):
    # pts=pts transformed via gt pose, poses=obj->cam
    pose_gt_inv = np.linalg.inv(pose_gt)
    # gt -> est
    if syms is not None:
        pose_ests = []
        for sym in syms:
            R = pose_est[:3, :3].dot(sym["R"])
            t = pose_est[:3, :3].dot(sym["t"].squeeze()) + pose_est[:3, 3]
            pose_ests.append(get_pose_mat_from_rt(R, t))
    else:
        pose_ests = [pose_est]

    all_dists = []
    for pose in pose_ests:
        delta_T = pose @ pose_gt_inv

        pts_est_transformed = misc.transform_pts_Rt(
            pts_gt_cam, delta_T[:3, :3], delta_T[:3, 3]
        )
        dists = np.linalg.norm(pts_est_transformed - pts_gt_cam, axis=1)
        all_dists.append(dists)
    best_pose_idx = np.argmin([d.mean() for d in all_dists])
    dists = all_dists[best_pose_idx]
    return dists


def plot_depth(
    depth,
    ax=None,
    include_colorbar=True,
    disable_axis=True,
    cmap="viridis",
    cbar_title=None,
    rgb=None,
    rgb_alpha=0.3,
    use_horiz_cbar=False,
    title=None,
    use_fixed_cbar=False,
    vmin=None,
    vmax=None,
    fontsize=14,
    use_white_bg=False,
    mask=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if disable_axis:
            ax.axis("off")
    kwargs = {}
    if use_fixed_cbar:
        kwargs.update({"vmin": vmin, "vmax": vmax})
    cmap = plt.get_cmap(cmap)
    cmap.set_bad("white")

    depth = adjust_depth_for_plt(depth)

    if use_white_bg:
        assert mask is not None
        depth = depth.copy()
        depth[~mask] = np.nan

    im = ax.imshow(depth, cmap=cmap, **kwargs)

    if include_colorbar:
        if cbar_title is None:
            cbar_title = "Depth (m)"
        if depth.max() > 100:
            cbar_title = cbar_title.replace("(m)", "(mm)")

        divider = make_axes_locatable(ax)
        if use_horiz_cbar:
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(cbar_title, fontsize=fontsize)
        else:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(cbar_title, rotation=90, labelpad=15, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    if rgb is not None:
        assert rgb.shape[:2] == depth.shape[:2], (rgb.shape, depth.shape)
        ax.imshow(rgb, alpha=rgb_alpha)

    if title:
        ax.set_title(title)

    return ax


def adjust_depth_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1:
        img = img.transpose(1, 2, 0)
    return img
