# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Abstract class of a renderer and a factory function to create a renderer.

The renderer produces an RGB/depth image of a 3D mesh model in a specified pose
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
        self.light_cam_pos = (0, 0, 0)

        # Set light color and weights.
        self.light_color = (1.0, 1.0, 1.0)  # Used only in C++ renderer.
        self.light_ambient_weight = 0.5
        self.light_diffuse_weight = 1.0  # Used only in C++ renderer.
        self.light_specular_weight = 0.0  # Used only in C++ renderer.
        self.light_specular_shininess = 0.0  # Used only in C++ renderer.

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

    def add_object(self, obj_id, model_path, **kwargs):
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

    def render_object(self, obj_id, R, t, fx, fy, cx, cy):
        """Renders an object model in the specified pose.

        :param obj_id: Object identifier.
        :param R: 3x3 ndarray with a rotation matrix.
        :param t: 3x1 ndarray with a translation vector.
        :param fx: Focal length (X axis).
        :param fy: Focal length (Y axis).
        :param cx: The X coordinate of the principal point.
        :param cy: The Y coordinate of the principal point.
        :return: Returns a dictionary with rendered images.
        """
        raise NotImplementedError


def create_renderer(
    width,
    height,
    renderer_type="cpp",
    mode="rgb+depth",
    shading="phong",
    bg_color=(0.0, 0.0, 0.0, 0.0),
):
    """A factory to create a renderer.

    Note: Parameters mode, shading and bg_color are currently supported only by
    the 'vispy' and 'python' renderers (renderer_type='vispy' or renderer_type='python').
    To render on a headless server, either 'vispy' or 'cpp' can be used.

    :param width: Width of the rendered image.
    :param height: Height of the rendered image.
    :param renderer_type: Type of renderer (options: 'vispy', 'cpp', 'python').
    :param mode: Rendering mode ('rgb+depth', 'rgb', 'depth').
    :param shading: Type of shading ('flat', 'phong').
    :param bg_color: Color of the background (R, G, B, A).
    :return: Instance of a renderer of the specified type.
    """
    if renderer_type == "python":
        from . import renderer_py

        return renderer_py.RendererPython(width, height, mode, shading, bg_color)
    elif renderer_type == "vispy":
        from . import renderer_vispy

        return renderer_vispy.RendererVispy(width, height, mode, shading, bg_color)
    elif renderer_type == "cpp":
        from . import renderer_cpp

        return renderer_cpp.RendererCpp(width, height)
    elif renderer_type == "htt":
        from . import renderer_htt

        return renderer_htt.RendererHtt(width, height)

    else:
        raise ValueError("Unknown renderer type.")
