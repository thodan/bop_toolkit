# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from bop_toolkit import inout, misc, renderer_py


def draw_rect(im, rect, color=(1.0, 1.0, 1.0)):
  """Draws a rectangle on an image.

  :param im: ndarray (uint8) on which the rectangle will be drawn.
  :param rect: Rectangle defined as [x, y, width, height], where [x, y] is the
    top-left corner.
  :param color: Color of the rectangle.
  :return: Image with drawn rectangle.
  """
  if im.dtype != np.uint8:
    raise ValueError('The image must be of type uint8.')

  im_pil = Image.fromarray(im)
  draw = ImageDraw.Draw(im_pil)
  draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                 outline=tuple([int(c * 255) for c in color]), fill=None)
  del draw
  return np.asarray(im_pil)


def write_text_on_image(im, txt_list, loc=(3, 0), color=(1.0, 1.0, 1.0),
                        size=20):
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
    font_path = os.path.join(os.path.dirname(__file__), 'droid_sans_mono.ttf')
    font = ImageFont.truetype(font_path, size)
  except IOError:
    misc.log('Warning: Loading a fallback font.')
    font = ImageFont.load_default()

  draw = ImageDraw.Draw(im_pil)
  for info in txt_list:
    if info['name'] != '':
      txt_tpl = '{}:{' + info['fmt'] + '}'
    else:
      txt_tpl = '{}{' + info['fmt'] + '}'
    txt = txt_tpl.format(info['name'], info['val'])
    draw.text(loc, txt, fill=tuple([int(c * 255) for c in color]), font=font)
    text_width, text_height = font.getsize(txt)
    loc = (loc[0], loc[1] + text_height)
  del draw

  return np.array(im_pil)


def vis_object_poses(
      poses, K, models, model_colors, rgb=None, depth=None, vis_rgb_path=None,
      vis_depth_diff_path=None, vis_orig_color=False,
      vis_rgb_resolve_visib=False):

  # Indicators of visualization types.
  vis_rgb = vis_rgb_path is not None
  vis_depth_diff = vis_depth_diff_path is not None

  if vis_rgb and rgb is None:
    raise ValueError('RGB visualization triggered but RGB image not provided.')

  if (vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib)) and depth is None:
    raise ValueError('Depth visualization triggered but D image not provided.')

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
        raise ValueError('The RGB and D images must have the same size.')
    else:
      im_size = (depth.shape[1], depth.shape[0])

  if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
    ren_depth = np.zeros((im_size[1], im_size[0]), np.float32)

  # Render the pose estimates one by one.
  for pose in poses:

    model = models[pose['obj_id']]
    model_color = model_colors[pose['obj_id']]

    # Rendering.
    m_rgb = None
    if vis_rgb:
      if vis_orig_color:
        m_rgb = renderer_py.render(
          model, im_size, K, pose['R'], pose['t'], mode='rgb')
      else:
        m_rgb = renderer_py.render(
          model, im_size, K, pose['R'], pose['t'], mode='rgb',
          surf_color=model_color)

    m_mask = None
    if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
      m_depth = renderer_py.render(
        model, im_size, K, pose['R'], pose['t'], mode='depth')

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

        if 'text_info' in pose:
          text_loc = (bbox[0] + 2, bbox[1])
          ren_rgb_info = write_text_on_image(
            ren_rgb_info, pose['text_info'], text_loc, color=text_color,
            size=text_size)

  # Blend and save the RGB visualization.
  if vis_rgb:
    vis_im_rgb = 0.5 * rgb.astype(np.float32) + \
                 0.5 * ren_rgb.astype(np.float32) + \
                 1.0 * ren_rgb_info.astype(np.float32)
    vis_im_rgb[vis_im_rgb > 255] = 255
    misc.ensure_dir(os.path.dirname(vis_rgb_path))
    inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8), jpg_quality=95)

  # Save the image of depth differences.
  if vis_depth_diff:
    # Calculate the depth difference at pixels where both depth maps
    # are valid.
    valid_mask = (depth > 0) * (ren_depth > 0)
    depth_diff = valid_mask * (depth - ren_depth.astype(np.float32))

    f, ax = plt.subplots(1, 1)
    cax = ax.matshow(depth_diff)
    ax.axis('off')
    ax.set_title('captured - GT depth [mm]')
    f.colorbar(cax, fraction=0.03, pad=0.01)
    f.tight_layout(pad=0)

    if not vis_rgb:
      misc.ensure_dir(os.path.dirname(vis_depth_diff_path))
    plt.savefig(vis_depth_diff_path, pad=0, bbox_inches='tight', quality=95)
    plt.close()
