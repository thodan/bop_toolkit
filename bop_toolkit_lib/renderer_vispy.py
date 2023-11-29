# Author: Gu Wang (guwang12@foxmail.com)
# Tsinghua University
# Adapted based on the glumpy version: "./renderer_py.py"
"""A Python Vispy based renderer."""

import inspect
import os
from typing import Hashable

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import OpenGL.GL as gl
import vispy
from bop_toolkit_lib import inout, misc, renderer
from vispy import app, gloo

# app backends: glfw, pyglet, egl
# gl backends: gl2, pyopengl2, gl+
app_backend = "egl"
gl_backend = "gl2"  # "pyopengl2"  # speed: 'gl+' < 'gl2' < 'pyopengl2'
vispy.use(app=app_backend, gl=gl_backend)
print("vispy uses app: {}, gl: {}".format(app_backend, gl_backend))

# RGB vertex shader.
_rgb_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_nm;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;
attribute vec2 a_texcoord;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_texcoord = a_texcoord;

    // The following points/vectors are expressed in the eye coordinates.
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex.
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light.
    v_normal = normalize(u_nm * vec4(a_normal, 1.0)).xyz; // Normal vector.
}
"""

# RGB fragment shader - flat shading.
_rgb_fragment_flat_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;

void main() {
    // Face normal in eye coords.
    vec3 f_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));

    float light_diffuse_w = max(dot(normalize(v_L), normalize(f_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture)) {
        gl_FragColor = vec4(light_w * texture2D(u_texture, v_texcoord));
    }
    else {
        gl_FragColor = vec4(light_w * v_color, 1.0);
    }
}
"""

# RGB fragment shader - Phong shading.
_rgb_fragment_phong_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    float light_diffuse_w = max(dot(normalize(v_L), normalize(v_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture)) {
        gl_FragColor = vec4(light_w * texture2D(u_texture, v_texcoord));
    }
    else {
        gl_FragColor = vec4(light_w * v_color, 1.0);
    }
}
"""

# Depth vertex shader.
# Ref: https://github.com/julienr/vertex_visibility/blob/master/depth.py
#
# Getting the depth from the depth buffer in OpenGL is doable, see here:
#   http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
#   http://web.archive.org/web/20130426093607/http://www.songho.ca/opengl/gl_projectionmatrix.html
#   http://stackoverflow.com/a/6657284/116067
# but it is difficult to achieve high precision, as explained in this article:
# http://dev.theomader.com/depth-precision/
#
# Once the vertex is in the view coordinates (view * model * v), its depth is
# simply the Z axis. Hence, instead of reading from the depth buffer and undoing
# the projection matrix, we store the Z coord of each vertex in the color
# buffer. OpenGL allows for float32 color buffer components.
_depth_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec3 a_color;
varying float v_eye_depth;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // In eye coords.

    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;
}
"""

# Depth fragment shader.
_depth_fragment_code = """
varying float v_eye_depth;

void main() {
    gl_FragColor = vec4(v_eye_depth, 0.0, 0.0, 1.0);
}
"""


# Functions to calculate transformation matrices.
# Note that OpenGL expects the matrices to be saved column-wise.
# (Ref: http://www.songho.ca/opengl/gl_transform.html)


def _calc_model_view(model, view):
    """Calculates the model-view matrix.

    :param model: 4x4 ndarray with the model matrix.
    :param view: 4x4 ndarray with the view matrix.
    :return: 4x4 ndarray with the model-view matrix.
    """
    return np.dot(model, view)


def _calc_model_view_proj(model, view, proj):
    """Calculates the model-view-projection matrix.

    :param model: 4x4 ndarray with the model matrix.
    :param view: 4x4 ndarray with the view matrix.
    :param proj: 4x4 ndarray with the projection matrix.
    :return: 4x4 ndarray with the model-view-projection matrix.
    """
    return np.dot(np.dot(model, view), proj)


def _calc_normal_matrix(model, view):
    """Calculates the normal matrix.

    Ref: http://www.songho.ca/opengl/gl_normaltransform.html

    :param model: 4x4 ndarray with the model matrix.
    :param view: 4x4 ndarray with the view matrix.
    :return: 4x4 ndarray with the normal matrix.
    """
    return np.linalg.inv(np.dot(model, view)).T


def _calc_calib_proj(K, x0, y0, w, h, nc, fc, window_coords="y_down"):
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    else:
        assert window_coords == "y_down"
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )
    return proj.T


class SingletonArgs(type):
    """Singleton that keep single instance for single set of arguments. E.g.:
    assert SingletonArgs('spam') is not SingletonArgs('eggs')
    assert SingletonArgs('spam') is SingletonArgs('spam')

    Source: https://gist.github.com/wowkin2/3af15bfbf197a14a2b0b2488a1e8c787
    """

    _instances = {}
    _init = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get("__init__", None)

    def __call__(cls, *args, **kwargs):
        def hashable(x):
            return x if isinstance(x, Hashable) else x.__str__()

        init = cls._init[cls]
        if init is not None:
            callargs_hashable = {
                hashable(k): hashable(v)
                for k, v in inspect.getcallargs(init, None, *args, **kwargs).items()
            }
            key = (cls, frozenset(callargs_hashable.items()))
        else:
            key = cls
        if key not in cls._instances:
            cls._instances[key] = super(SingletonArgs, cls).__call__(*args, **kwargs)
        return cls._instances[key]


class RendererVispy(renderer.Renderer, app.Canvas, metaclass=SingletonArgs):
    """A Python based renderer."""

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
        renderer.Renderer.__init__(self, width=width, height=height)
        app.Canvas.__init__(self, show=False, size=(width, height))

        self.mode = mode
        self.shading = shading
        self.bg_color = bg_color

        # yz flip: opencv to opengl
        pose_cv_to_gl = np.eye(4, dtype=np.float32)
        pose_cv_to_gl[1, 1], pose_cv_to_gl[2, 2] = -1, -1
        self.pose_cv_to_gl = pose_cv_to_gl

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

        # Per-object vertex and index buffer.
        self.vertex_buffers = {}
        self.index_buffers = {}

        # Per-object OpenGL programs for rendering of RGB and depth images.
        self.rgb_programs = {}
        self.depth_programs = {}

        # The frame buffer object.
        rgb_buf = gloo.Texture2D(shape=(self.height, self.width, 3))
        depth_buf = gloo.RenderBuffer(shape=(self.height, self.width))
        self.fbo = gloo.FrameBuffer(color=rgb_buf, depth=depth_buf)
        # Activate the created frame buffer object.
        self.fbo.activate()

    def add_object(self, obj_id, model_path, **kwargs):
        """See base class."""
        # Color of the object model (the original color saved with the object model
        # will be used if None).
        surf_color = None
        if "surf_color" in kwargs:
            surf_color = kwargs["surf_color"]

        # Load the object model.
        model = inout.load_ply(model_path)
        self.models[obj_id] = model

        # Calculate the 3D bounding box of the model (will be used to set the near
        # and far clipping plane).
        bb = misc.calc_3d_bbox(
            model["pts"][:, 0], model["pts"][:, 1], model["pts"][:, 2]
        )
        self.model_bbox_corners[obj_id] = np.array(
            [
                [bb[0], bb[1], bb[2]],
                [bb[0], bb[1], bb[2] + bb[5]],
                [bb[0], bb[1] + bb[4], bb[2]],
                [bb[0], bb[1] + bb[4], bb[2] + bb[5]],
                [bb[0] + bb[3], bb[1], bb[2]],
                [bb[0] + bb[3], bb[1], bb[2] + bb[5]],
                [bb[0] + bb[3], bb[1] + bb[4], bb[2]],
                [bb[0] + bb[3], bb[1] + bb[4], bb[2] + bb[5]],
            ]
        )

        # Set texture/color of vertices.
        self.model_textures[obj_id] = None

        # Use the specified uniform surface color.
        if surf_color is not None:
            colors = np.tile(list(surf_color) + [1.0], [model["pts"].shape[0], 1])

            # Set UV texture coordinates to dummy values.
            texture_uv = np.zeros((model["pts"].shape[0], 2), np.float32)

        # Use the model texture.
        elif "texture_file" in self.models[obj_id].keys():
            model_texture_path = os.path.join(
                os.path.dirname(model_path), self.models[obj_id]["texture_file"]
            )
            model_texture = inout.load_im(model_texture_path)

            # Normalize the texture image.
            if model_texture.max() > 1.0:
                model_texture = model_texture.astype(np.float32) / 255.0
            model_texture = np.flipud(model_texture)
            self.model_textures[obj_id] = model_texture

            # UV texture coordinates.
            texture_uv = model["texture_uv"]

            # Set the per-vertex color to dummy values.
            colors = np.zeros((model["pts"].shape[0], 3), np.float32)

        # Use the original model color.
        elif "colors" in model.keys():
            assert model["pts"].shape[0] == model["colors"].shape[0]
            colors = model["colors"]
            if colors.max() > 1.0:
                colors /= 255.0  # Color values are expected in range [0, 1].

            # Set UV texture coordinates to dummy values.
            texture_uv = np.zeros((model["pts"].shape[0], 2), np.float32)

        # Set the model color to gray.
        else:
            colors = np.ones((model["pts"].shape[0], 3), np.float32) * 0.5

            # Set UV texture coordinates to dummy values.
            texture_uv = np.zeros((model["pts"].shape[0], 2), np.float32)

        # Set the vertex data.
        if self.mode == "depth":
            vertices_type = [
                ("a_position", np.float32, 3),
                ("a_color", np.float32, colors.shape[1]),
            ]
            vertices = np.array(list(zip(model["pts"], colors)), vertices_type)
        else:
            if self.shading == "flat":
                vertices_type = [
                    ("a_position", np.float32, 3),
                    ("a_color", np.float32, colors.shape[1]),
                    ("a_texcoord", np.float32, 2),
                ]
                vertices = np.array(
                    list(zip(model["pts"], colors, texture_uv)), vertices_type
                )
            elif self.shading == "phong":
                vertices_type = [
                    ("a_position", np.float32, 3),
                    ("a_normal", np.float32, 3),
                    ("a_color", np.float32, colors.shape[1]),
                    ("a_texcoord", np.float32, 2),
                ]
                vertices = np.array(
                    list(zip(model["pts"], model["normals"], colors, texture_uv)),
                    vertices_type,
                )
            else:
                raise ValueError("Unknown shading type.")

        # Create vertex and index buffer for the loaded object model.
        self.vertex_buffers[obj_id] = gloo.VertexBuffer(vertices)
        self.index_buffers[obj_id] = gloo.IndexBuffer(
            model["faces"].flatten().astype(np.uint32)
        )

        # Set shader for the selected shading.
        if self.shading == "flat":
            rgb_fragment_code = _rgb_fragment_flat_code
        elif self.shading == "phong":
            rgb_fragment_code = _rgb_fragment_phong_code
        else:
            raise ValueError("Unknown shading type.")

        # Prepare the RGB OpenGL program.
        rgb_program = gloo.Program(_rgb_vertex_code, rgb_fragment_code)
        rgb_program.bind(self.vertex_buffers[obj_id])
        if self.model_textures[obj_id] is not None:
            rgb_program["u_use_texture"] = int(True)
            rgb_program["u_texture"] = self.model_textures[obj_id]
        else:
            rgb_program["u_use_texture"] = int(False)
            rgb_program["u_texture"] = np.zeros((1, 1, 4), np.float32)
        self.rgb_programs[obj_id] = rgb_program

        # Prepare the depth OpenGL program.
        depth_program = gloo.Program(_depth_vertex_code, _depth_fragment_code)
        depth_program.bind(self.vertex_buffers[obj_id])
        self.depth_programs[obj_id] = depth_program

    def remove_object(self, obj_id):
        """See base class."""
        del self.models[obj_id]
        del self.model_bbox_corners[obj_id]
        if obj_id in self.model_textures:
            del self.model_textures[obj_id]
        del self.vertex_buffers[obj_id]
        del self.index_buffers[obj_id]
        del self.rgb_programs[obj_id]
        del self.depth_programs[obj_id]

    def render_object(self, obj_id, R, t, fx, fy, cx, cy, clear=True):
        """See base class."""

        # Model matrix (from object space to world space).
        mat_model = np.eye(4, dtype=np.float32)

        # View matrix (from world space to eye space; transforms also the coordinate
        # system from OpenCV to OpenGL camera space).
        mat_view_cv = np.eye(4, dtype=np.float32)
        mat_view_cv[:3, :3], mat_view_cv[:3, 3] = R, t.squeeze()

        mat_view = self.pose_cv_to_gl.dot(
            mat_view_cv
        )  # OpenCV to OpenGL camera system.
        mat_view = mat_view.T  # OpenGL expects column-wise matrix format.

        # Calculate the near and far clipping plane from the 3D bounding box.
        bbox_corners = self.model_bbox_corners[obj_id]
        bbox_corners_ht = np.concatenate(
            (bbox_corners, np.ones((bbox_corners.shape[0], 1))), axis=1
        ).transpose()
        bbox_corners_eye_z = mat_view_cv[2, :].reshape((1, 4)).dot(bbox_corners_ht)
        self.clip_near = bbox_corners_eye_z.min()
        self.clip_far = bbox_corners_eye_z.max()

        # Projection matrix.
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        mat_proj = _calc_calib_proj(
            K, 0, 0, self.width, self.height, self.clip_near, self.clip_far
        )

        self.update()
        self.on_draw(obj_id, mat_model, mat_view, mat_proj, clear=clear)

        if self.mode == "rgb":
            return {"rgb": self.rgb}
        elif self.mode == "depth":
            return {"depth": self.depth}
        elif self.mode == "rgb+depth":
            return {"rgb": self.rgb, "depth": self.depth}

    def on_draw(self, obj_id, mat_model, mat_view, mat_proj, clear=True):
        with self.fbo:
            gloo.set_state(depth_test=True, blend=False, cull_face=False)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            # gl.glDisable(gl.GL_LINE_SMOOTH)
            if clear:
                gloo.set_clear_color(
                    (
                        self.bg_color[0],
                        self.bg_color[1],
                        self.bg_color[2],
                        self.bg_color[3],
                    )
                )
                gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, self.width, self.height)

            if self.render_rgb:
                self.rgb = self._draw_rgb(obj_id, mat_model, mat_view, mat_proj)

            if self.render_depth:
                self.depth = self._draw_depth(obj_id, mat_model, mat_view, mat_proj)

    def _draw_rgb(self, obj_id, mat_model, mat_view, mat_proj):
        """Renders an RGB image.

        :param obj_id: ID of the object model to render.
        :param mat_model: 4x4 ndarray with the model matrix.
        :param mat_view: 4x4 ndarray with the view matrix.
        :param mat_proj: 4x4 ndarray with the projection matrix.
        :return: HxWx3 ndarray with the rendered RGB image.
        """
        # Update the OpenGL program.
        program = self.rgb_programs[obj_id]
        program["u_light_eye_pos"] = list(self.light_cam_pos)
        program["u_light_ambient_w"] = self.light_ambient_weight
        program["u_mv"] = _calc_model_view(mat_model, mat_view)
        program["u_nm"] = _calc_normal_matrix(mat_model, mat_view)
        program["u_mvp"] = _calc_model_view_proj(mat_model, mat_view, mat_proj)

        # Rendering.
        program.draw("triangles", self.index_buffers[obj_id])

        # Get the content of the FBO texture.
        rgb = gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        rgb = np.frombuffer(rgb, dtype=np.uint8).reshape((self.height, self.width, 3))[
            ::-1, :
        ]
        return rgb

    def _draw_depth(self, obj_id, mat_model, mat_view, mat_proj):
        """Renders a depth image.

        :param obj_id: ID of the object model to render.
        :param mat_model: 4x4 ndarray with the model matrix.
        :param mat_view: 4x4 ndarray with the view matrix.
        :param mat_proj: 4x4 ndarray with the projection matrix.
        :return: HxW ndarray with the rendered depth image.
        """
        # Update the OpenGL program.
        program = self.depth_programs[obj_id]
        program["u_mv"] = _calc_model_view(mat_model, mat_view)
        program["u_mvp"] = _calc_model_view_proj(mat_model, mat_view, mat_proj)

        # Rendering.
        program.draw("triangles", self.index_buffers[obj_id])

        dep = gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT
        )
        # self.depth = self.depth.reshape(self.height, self.width)
        # Read buffer and flip X
        dep = np.copy(np.frombuffer(dep, np.float32)).reshape(self.height, self.width)[
            ::-1, :
        ]

        # Convert z-buffer to depth map
        mult = (self.clip_near * self.clip_far) / (self.clip_near - self.clip_far)
        addi = self.clip_far / (self.clip_near - self.clip_far)
        bg = dep == 1
        dep = mult / (dep + addi)
        dep[bg] = 0

        return dep
