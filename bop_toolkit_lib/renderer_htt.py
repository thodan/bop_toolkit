
# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""A wrapper around Hand Tracking Toolkit rasterizer."""

import trimesh

from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer

from hand_tracking_toolkit.rasterizer import rasterize_mesh
from hand_tracking_toolkit.camera import CameraModel


def subdivide_mesh(
    mesh: trimesh.Trimesh,
    max_edge: float = 0.005,
    max_iters: int = 50,
    debug: bool = False,
):
    """Subdivides mesh such as all edges are shorter than a threshold.

    Args:
        mesh: Mesh to subdivide.
        max_edge: Maximum allowed edge length in meters (note that this may
            not be reachable if max_iters is too low).
        max_iters: Number of subdivision iterations.
    Returns.
        Subdivided mesh.
    """

    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices,
        mesh.faces,
        max_edge,
        max_iter=max_iters,
    )
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    if debug:
        print(f"Remeshing: {len(mesh.vertices)} -> {len(new_mesh.vertices)}")

    return new_mesh



class RendererHtt(renderer.Renderer):
    """A wrapper around Hand Tracking Toolkit rasterizer."""

    def __init__(
        self,
        width,
        height,
        mode=None,
        shading="phong",
        bg_color=(0.0, 0.0, 0.0, 0.0),
    ):
        """Constructor.

        :param width: Width of the rendered image.
        :param height: Height of the rendered image.
        :param mode: Kept for consistency with Render API, htt render always render rgb+depth+mask.
        :param shading: Type of shading ('flat', 'phong').
        :param bg_color: Color of the background (R, G, B, A).
        """
        super(RendererHtt, self).__init__(width, height)

        self.mode = mode
        self.shading = shading
        self.bg_color = bg_color

        # Structures to store object models and related info.
        self.models = {}

    def add_object(self, obj_id, model_path, **kwargs):
        """See base class."""
        model = trimesh.load(model_path)
        # Make sure there are no large triangles (the rasterizer
        # from hand_tracking_toolkit becomes slow if some triangles
        # are much larger than others)
        model = subdivide_mesh(model, max_edge=5.0) 
        self.models[obj_id] = model

    def remove_object(self, obj_id):
        """See base class."""
        del self.models[obj_id]

    def render_object(self, obj_id, R, t, camera: CameraModel):
        """See base class."""

        # transform points to camera frame
        pts = misc.transform_pts_Rt(self.models[obj_id].vertices, R, t)
        rgb, mask, depth = rasterize_mesh(pts, self.models[obj_id].faces, camera)

        return {"rgb": rgb, "mask": mask, "depth": depth}
