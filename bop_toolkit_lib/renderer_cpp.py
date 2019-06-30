# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""An interface to the C++ based renderer (bop_renderer)."""

import sys

from bop_toolkit_lib import config
from bop_toolkit_lib import renderer

# C++ renderer (https://github.com/thodan/bop_renderer)
sys.path.append(config.bop_renderer_path)
import bop_renderer


class RendererCpp(renderer.Renderer):
  """An interface to the C++ based renderer."""

  def __init__(self, width, height):
    """See base class."""
    super(RendererCpp, self).__init__(width, height)
    self.renderer = bop_renderer.PyRenderer()
    self.renderer.init(width, height)

  def set_light_cam_pos(self, light_cam_pos):
    """See base class."""
    super(RendererCpp, self).set_light_cam_pos(light_cam_pos)
    self.renderer.set_light_cam_pos(light_cam_pos)

  def set_light_ambient_weight(self, light_ambient_weight):
    """See base class."""
    super(RendererCpp, self).set_light_ambient_weight(light_ambient_weight)
    self.renderer.set_light_ambient_weight(light_ambient_weight)

  def add_object(self, obj_id, model_path, **kwargs):
    """See base class."""
    self.renderer.add_object(obj_id, model_path)

  def remove_object(self, obj_id):
    """See base class."""
    self.renderer.remove_object(obj_id)

  def render_object(self, obj_id, R, t, fx, fy, cx, cy):
    """See base class."""
    R_l = map(float, R.flatten().tolist())
    t_l = map(float, t.flatten().tolist())
    self.renderer.render_object(obj_id, R_l, t_l, fx, fy, cx, cy)
    rgb = self.renderer.get_color_image(obj_id)
    depth = self.renderer.get_depth_image(obj_id)
    return {'rgb': rgb, 'depth': depth}
