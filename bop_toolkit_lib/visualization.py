# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualization utilities."""

import os

# import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

htt_available = False
try:
    from hand_tracking_toolkit.camera import CameraModel
    htt_available = True
except ImportError as e:
    logger.warn("""Missing hand_tracking_toolkit dependency,
                mandatory if you are running evaluation on HOT3d.
                Refer to the README.md for installation instructions.
                """)


def draw_rect(im, rect, color=(1.0, 1.0, 1.0)):
    """Draws a rectangle on an image.

    :param im: ndarray (uint8) on which the rectangle will be drawn.
    :param rect: Rectangle defined as [x, y, width, height], where [x, y] is the
      top-left corner.
    :param color: Color of the rectangle.
    :return: Image with drawn rectangle.
    """
    if im.dtype != np.uint8:
        raise ValueError("The image must be of type uint8.")

    im_pil = Image.fromarray(im)
    draw = ImageDraw.Draw(im_pil)
    draw.rectangle(
        (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
        outline=tuple([int(c * 255) for c in color]),
        fill=None,
    )
    del draw
    return np.asarray(im_pil)


def write_text_on_image(im, txt_list, loc=(3, 0), color=(1.0, 1.0, 1.0), size=20):
    """Writes text info on an image.

    :param im: ndarray on which the text info will be written.
    :param txt_list: List of dictionaries, each describing one info line:
      - 'name': Entry name.
      - 'val': Entry value.
      - 'fmt': String format for the value.
    :param loc: Location of the top left corner of the text box.
    :param color: Font color.
    :param size: Font size.
    :return: Image with written text info.
    """
    im_pil = Image.fromarray(im)

    # Load font.
    try:
        font_path = os.path.join(os.path.dirname(__file__), "droid_sans_mono.ttf")
        font = ImageFont.truetype(font_path, size)
    except IOError:
        misc.log("Warning: Loading a fallback font.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(im_pil)
    for info in txt_list:
        if info["name"] != "":
            txt_tpl = "{}:{" + info["fmt"] + "}"
        else:
            txt_tpl = "{}{" + info["fmt"] + "}"
        txt = txt_tpl.format(info["name"], info["val"])
        draw.text(loc, txt, fill=tuple([int(c * 255) for c in color]), font=font)
        text_width, text_height = font.getsize(txt)
        loc = (loc[0], loc[1] + text_height)
    del draw

    return np.array(im_pil)


def depth_for_vis(depth, valid_start=0.2, valid_end=1.0):
    """Transforms depth values from the specified range to [0, 255].

    :param depth: ndarray with a depth image (1 channel).
    :param valid_start: The beginning of the depth range.
    :param valid_end: The end of the depth range.
    :return: Transformed depth image.
    """
    mask = depth > 0
    depth_n = depth.astype(np.float64)
    depth_n[mask] -= depth_n[mask].min()
    depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
    depth_n[mask] += valid_start
    return depth_n


def vis_object_poses(
    poses,
    K,
    renderer,
    rgb=None,
    depth=None,
    vis_rgb_path=None,
    vis_depth_diff_path=None,
    vis_rgb_resolve_visib=False,
):
    """Visualizes 3D object models in specified poses in a single image.

    Two visualizations are created:
    1. An RGB visualization (if vis_rgb_path is not None).
    2. A Depth-difference visualization (if vis_depth_diff_path is not None).

    :param poses: List of dictionaries, each with info about one pose:
      - 'obj_id': Object ID.
      - 'R': 3x3 ndarray with a rotation matrix.
      - 't': 3x1 ndarray with a translation vector.
      - 'text_info': Info to write at the object (see write_text_on_image).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param rgb: ndarray with the RGB input image.
    :param depth: ndarray with the depth input image.
    :param vis_rgb_path: Path to the output RGB visualization.
    :param vis_depth_diff_path: Path to the output depth-difference visualization.
    :param vis_rgb_resolve_visib: Whether to resolve visibility of the objects
      (i.e. only the closest object is visualized at each pixel).
    """

    # Indicators of visualization types.
    vis_rgb = vis_rgb_path is not None
    vis_depth_diff = vis_depth_diff_path is not None

    if vis_rgb and rgb is None:
        raise ValueError("RGB visualization triggered but RGB image not provided.")

    if (vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib)) and depth is None:
        raise ValueError("Depth visualization triggered but D image not provided.")

    # Prepare images for rendering.
    im_size = None
    ren_rgb = None
    ren_rgb_info = None
    ren_depth = None

    if vis_rgb:
        im_size = (rgb.shape[1], rgb.shape[0])
        ren_rgb = np.zeros(rgb.shape, np.uint8)
        ren_rgb_info = np.zeros(rgb.shape, np.uint8)

    if vis_depth_diff:
        if im_size and im_size != (depth.shape[1], depth.shape[0]):
            raise ValueError("The RGB and D images must have the same size.")
        else:
            im_size = (depth.shape[1], depth.shape[0])

    if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
        ren_depth = np.zeros((im_size[1], im_size[0]), np.float32)

    # Render the pose estimates one by one.
    for pose in poses:
        # Rendering.
        if htt_available and isinstance(K, CameraModel): # hand_tracking_toolkit is used for rendering.
            ren_out = renderer.render_object(
                pose["obj_id"], pose["R"], pose["t"], K
            )
        elif isinstance(K, np.ndarray) and K.shape == (3, 3):  # pinhole camera model is used for rendering.
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            ren_out = renderer.render_object(
                pose["obj_id"], pose["R"], pose["t"], fx, fy, cx, cy
            )
        else:
            raise ValueError("Camera model 'K' type {} should be either CameraModel or np.ndarray".format(type(K)))

        m_rgb = None
        if vis_rgb:
            m_rgb = ren_out["rgb"]

        m_mask = None
        if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
            m_depth = ren_out["depth"]

            # Get mask of the surface parts that are closer than the
            # surfaces rendered before.
            visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
            m_mask = np.logical_and(m_depth != 0, visible_mask)

            ren_depth[m_mask] = m_depth[m_mask].astype(ren_depth.dtype)

        # Combine the RGB renderings.
        if vis_rgb:
            if vis_rgb_resolve_visib:
                ren_rgb[m_mask] = m_rgb[m_mask].astype(ren_rgb.dtype)
            else:
                ren_rgb_f = ren_rgb.astype(np.float32) + m_rgb.astype(np.float32)
                ren_rgb_f[ren_rgb_f > 255] = 255
                ren_rgb = ren_rgb_f.astype(np.uint8)

            # Draw 2D bounding box and write text info.
            obj_mask = np.sum(m_rgb > 0, axis=2)
            ys, xs = obj_mask.nonzero()
            if len(ys):
                # bbox_color = model_color
                # text_color = model_color
                bbox_color = (0.3, 0.3, 0.3)
                text_color = (1.0, 1.0, 1.0)
                text_size = 11

                bbox = misc.calc_2d_bbox(xs, ys, im_size)
                im_size = (obj_mask.shape[1], obj_mask.shape[0])
                ren_rgb_info = draw_rect(ren_rgb_info, bbox, bbox_color)

                if "text_info" in pose:
                    text_loc = (bbox[0] + 2, bbox[1])
                    ren_rgb_info = write_text_on_image(
                        ren_rgb_info,
                        pose["text_info"],
                        text_loc,
                        color=text_color,
                        size=text_size,
                    )

    # Blend and save the RGB visualization.
    if vis_rgb:
        misc.ensure_dir(os.path.dirname(vis_rgb_path))

        vis_im_rgb = (
            0.5 * rgb.astype(np.float32)
            + 0.5 * ren_rgb.astype(np.float32)
            + 1.0 * ren_rgb_info.astype(np.float32)
        )
        vis_im_rgb[vis_im_rgb > 255] = 255
        inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8), jpg_quality=95)

    # Save the image of depth differences.
    if vis_depth_diff:
        misc.ensure_dir(os.path.dirname(vis_depth_diff_path))

        # Calculate the depth difference at pixels where both depth maps are valid.
        valid_mask = (depth > 0) * (ren_depth > 0)
        depth_diff = valid_mask * (ren_depth.astype(np.float32) - depth)

        # Get mask of pixels where the rendered depth is at most by the tolerance
        # delta behind the captured depth (this tolerance is used in VSD).
        delta = 15
        below_delta = valid_mask * (depth_diff < delta)
        below_delta_vis = (255 * below_delta).astype(np.uint8)

        depth_diff_vis = 255 * depth_for_vis(depth_diff - depth_diff.min())

        # Pixels where the rendered depth is more than the tolerance delta behing
        # the captured depth will be cyan.
        depth_diff_vis = np.dstack(
            [below_delta_vis, depth_diff_vis, depth_diff_vis]
        ).astype(np.uint8)

        depth_diff_vis[np.logical_not(valid_mask)] = 0
        depth_diff_valid = depth_diff[valid_mask]
        depth_info = [
            {"name": "min diff", "fmt": ":.3f", "val": np.min(depth_diff_valid)},
            {"name": "max diff", "fmt": ":.3f", "val": np.max(depth_diff_valid)},
            {"name": "mean diff", "fmt": ":.3f", "val": np.mean(depth_diff_valid)},
            {"name": "median diff", "fmt": ":.3f", "val": np.median(np.abs(depth_diff_valid))},
            {"name": "25 percentile", "fmt": ":.3f", "val": np.percentile(np.abs(depth_diff_valid), 25)},
        ]
        depth_diff_vis = write_text_on_image(depth_diff_vis, depth_info)
        inout.save_im(vis_depth_diff_path, depth_diff_vis)
