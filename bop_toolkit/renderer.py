# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Abstract class of a renderer and a factory function to create a renderer

The renderer produces an rgb/depth image of a 3D mesh model in a specified pose
for given camera parameters and illumination settings.
"""


class Renderer(object):
  """Abstract class of a renderer."""

  def __init__(self, width, height):
    """Constructor.

    :param width: Width of the rendered image.
    :param height: Height of the rendered image.
    """
    self.width = width
    self.height = height

    # 3D location of a point light (in the camera coordinates).
    self.light_cam_pos = [0, 0, 0]

    # Weight of the ambient light.
    self.light_ambient_weight = 0.5

  def set_light_cam_pos(self, light_cam_pos):
    """Sets the 3D location of a point light.

    :param light_cam_pos: [X, Y, Z].
    """
    self.light_cam_pos = light_cam_pos

  def set_light_ambient_weight(self, light_ambient_weight):
    """Sets weight of the ambient light.

    :param light_ambient_weight: Scalar from 0 to 1.
    """
    self.light_ambient_weight = light_ambient_weight

  def add_object(self, obj_id, model_path):
    """Loads an object model.

    :param obj_id: Object identifier.
    :param model_path: Path to the object model file.
    """
    raise NotImplementedError

  def remove_object(self, obj_id):
    """Removes an object model.

    :param obj_id: Identifier of the object to remove.
    """
    raise NotImplementedError

  def render_object(self, obj_id, R, t, fx, fy, cx, cy, shading='flat'):
    """Renders an object model in the specified pose.

    :param obj_id: Object identifier.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :param fx: Focal length (X axis).
    :param fy: Focal length (Y axis).
    :param cx: The X coordinate of the principal point.
    :param cy: The Y coordinate of the principal point.
    :param shading: Type of shading.
    """
    raise NotImplementedError

  def get_color_image(self, obj_id):
    """Returns the last rendered RGB image of the specified object.

    :param obj_id: Object identifier.
    """
    raise NotImplementedError

  def get_depth_image(self, obj_id):
    """Returns the last rendered depth image of the specified object.

    :param obj_id: Object identifier.
    """
    raise NotImplementedError


def create_renderer(im_size, renderer_type='cpp'):
  """A factory to create a renderer.

  :param im_size: Size (width, height) of the rendered image.
  :param renderer_type: Type of renderer (options: 'cpp', 'python').
  :return: Instance of a renderer of the specified type.
  """
  if renderer_type == 'python':
    from . import renderer_py
    return renderer_py.RendererPython(im_size[0], im_size[1])

  elif renderer_type == 'cpp':
    from . import renderer_cpp
    return renderer_cpp.RendererCpp(im_size[0], im_size[1])

  else:
    raise ValueError('Unknown renderer type.')
