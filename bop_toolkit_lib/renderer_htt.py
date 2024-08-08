
# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Center for Machine Perception, Czech Technical University in Prague

"""A wrapper around Hand Tracking Toolkit rasterizer."""

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer

from hand_tracking_toolkit.rasterizer import rasterize_mesh
from hand_tracking_toolkit.camera import CameraModel

class RendererHtt(renderer.Renderer):
    """A wrapper around Hand Tracking Toolkit rasterizer."""

    def __init__(
        self,
        width,
        height,
        mode="rgb+depth",
        shading="phong",
        bg_color=(0.0, 0.0, 0.0, 0.0),
    ):
        """Constructor.

        :param width: Width of the rendered image.
        :param height: Height of the rendered image.
        :param mode: Rendering mode ('rgb+depth', 'rgb', 'depth').
        :param shading: Type of shading ('flat', 'phong').
        :param bg_color: Color of the background (R, G, B, A).
        """
        super(RendererHtt, self).__init__(width, height)

        self.mode = mode
        self.shading = shading
        self.bg_color = bg_color

        # Indicators whether to render RGB and/or depth image.
        self.render_rgb = self.mode in ["rgb", "rgb+depth"]
        self.render_depth = self.mode in ["depth", "rgb+depth"]

        # Structures to store object models and related info.
        self.models = {}
        self.model_bbox_corners = {}
        self.model_textures = {}

        # Rendered images.
        self.rgb = None
        self.depth = None

    def add_object(self, obj_id, model_path, **kwargs):
        """See base class."""
        # Color of the object model (the original color saved with the object model
        # will be used if None).

        # Load the object model.
        model = inout.load_ply(model_path)
        self.models[obj_id] = model

    def remove_object(self, obj_id):
        """See base class."""
        del self.models[obj_id]

    def render_object(self, obj_id, R, t, camera: CameraModel):
        """See base class."""

        # transform points to camera frame
        pts_c = misc.transform_pts_Rt(self.models[obj_id]["pts"], R, t)

        self.rgb, self.mask, self.depth = rasterize_mesh(pts_c, self.models[obj_id]["faces"], camera)

        if self.mode == "rgb":
            return {"rgb": self.rgb}
        elif self.mode == "depth":
            return {"depth": self.depth}
        elif self.mode == "rgb+depth":
            return {"rgb": self.rgb, "depth": self.depth}
