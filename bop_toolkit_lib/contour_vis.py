
import cv2
import numpy as np


def draw_pose_contour(
    cv_img,
    rendered_depth,
    contour_color=(255, 0, 0),
    thickness=3,
    mask_visib=None,
):
    # based on: https://github.com/megapose6d/megapose6d/blob/master/src/megapose/visualization/utils.py

    mask = ((rendered_depth > 0).astype(np.uint8)) * 255

    if mask_visib is not None:
        mask = mask * mask_visib

    # clean up small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, k, iterations=1)
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
