import os

import cv2
import numpy as np


def draw_pose_contour(
    cv_img,
    K,
    obj_pose,
    mesh=None,
    contour_color=(255, 0, 0),
    thickness=3,
    headless=False,
    mask_visib=None,
    render_out=None,
):
    # based on: https://github.com/megapose6d/megapose6d/blob/master/src/megapose/visualization/utils.py

    if render_out is None:
        assert mesh is not None
        rendered_color, depth = render_offscreen(
            mesh,
            obj_pose,
            K,
            w=cv_img.shape[1],
            h=cv_img.shape[0],
            headless=headless,
        )
    else:
        rendered_color, depth = render_out["color"], render_out["depth"]

    mask = ((depth > 0).astype(np.uint8)) * 255

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

    # overlay
    if cv_img.ndim == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    cv_img[edge > 0] = contour_color

    return rendered_color, cv_img


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


def render_offscreen(mesh, obj_pose, intrinsic, w, h, headless=False):
    import pyrender

    if headless:
        os.environ["DISPLAY"] = ":1"
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    cam_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    scene = pyrender.Scene(
        bg_color=np.array([1, 1, 1, 0]), ambient_light=np.array([0.2, 0.2, 0.2, 1.0])
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=4.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100000
    )

    scene.add(light, pose=cam_pose)
    scene.add(camera, pose=cam_pose)

    # If you actually have a mesh, prefer from_trimesh; from_points renders a point cloud.
    if isinstance(mesh, dict) and "pts" in mesh:
        mesh_node = pyrender.Mesh.from_points(mesh["pts"])
    else:
        mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node, pose=obj_pose)

    r = pyrender.OffscreenRenderer(w, h)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.OFFSCREEN)
    r.delete()
    return color, depth
