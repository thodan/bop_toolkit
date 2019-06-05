# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""A Python based renderer."""

import os
import numpy as np
from glumpy import app, gloo, gl

from bop_toolkit import renderer, inout

# Set glumpy logging level.
from glumpy.log import log
import logging
log.setLevel(logging.WARNING)  # Options: ERROR, WARNING, DEBUG, INFO.

# Set backend (http://glumpy.readthedocs.io/en/latest/api/app-backends.html).
# app.use('glfw') # Options: 'glfw', 'qt5', 'pyside', 'pyglet'


# Color vertex shader.
_color_vertex_code = """
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

# Color fragment shader - flat shading.
_color_fragment_flat_code = """
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

# Color fragment shader - Phong shading.
_color_fragment_phong_code = """
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


def _calc_calib_proj(K, x0, y0, w, h, nc, fc, window_coords='y_down'):
  """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  :param K: 3x3 ndarray with the camera matrix.
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
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
    ])

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  else:
    assert window_coords == 'y_down'
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
    ])
  return proj.T


def draw_color(shape, vertex_buffer, index_buffer, texture, mat_model, mat_view,
               mat_proj, ambient_weight, bg_color, shading):
  """Renders an RGB image.

  :param shape: Shape (H, W) of the image to render.
  :param vertex_buffer: Vertex buffer object.
  :param index_buffer: Index buffer object.
  :param texture: MxNx3 ndarray with the object model texture.
  :param mat_model: 4x4 ndarray with the model matrix.
  :param mat_view: 4x4 ndarray with the view matrix.
  :param mat_proj: 4x4 ndarray with the projection matrix.
  :param ambient_weight: Weight of the ambient light.
  :param bg_color: Color of the background (R, G, B, A).
  :param shading: Type of shading ('flat' or 'phong').
  :return: HxWx3 ndarray with the rendered RGB image.
  """
  # Set shader for the selected shading.
  if shading == 'flat':
    color_fragment_code = _color_fragment_flat_code
  elif shading == 'phong':
    color_fragment_code = _color_fragment_phong_code
  else:
    raise ValueError('Unknown shading type.')

  # Prepare the OpenGL program.
  program = gloo.Program(_color_vertex_code, color_fragment_code)
  program.bind(vertex_buffer)
  program['u_light_eye_pos'] = [0, 0, 0]  # Camera origin.
  program['u_light_ambient_w'] = ambient_weight
  program['u_mv'] = _calc_model_view(mat_model, mat_view)
  program['u_nm'] = _calc_normal_matrix(mat_model, mat_view)
  program['u_mvp'] = _calc_model_view_proj(mat_model, mat_view, mat_proj)
  if texture is not None:
    program['u_use_texture'] = int(True)
    program['u_texture'] = texture
  else:
    program['u_use_texture'] = int(False)
    program['u_texture'] = np.zeros((1, 1, 4), np.float32)

  # Frame buffer object.
  color_buf = np.zeros(
    (shape[0], shape[1], 4), np.float32).view(gloo.TextureFloat2D)
  depth_buf = np.zeros(
    (shape[0], shape[1]), np.float32).view(gloo.DepthTexture)
  fbo = gloo.FrameBuffer(color=color_buf, depth=depth_buf)
  fbo.activate()

  # OpenGL setup.
  gl.glEnable(gl.GL_DEPTH_TEST)
  gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
  gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
  gl.glViewport(0, 0, shape[1], shape[0])
  # gl.glEnable(gl.GL_BLEND)
  # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
  # gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
  # gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
  # gl.glDisable(gl.GL_LINE_SMOOTH)
  # gl.glDisable(gl.GL_POLYGON_SMOOTH)
  # gl.glEnable(gl.GL_MULTISAMPLE)

  # Keep the back-face culling disabled because of objects which do not have
  # well-defined surface (e.g. the lamp from the lm dataset).
  gl.glDisable(gl.GL_CULL_FACE)
  # gl.glEnable(gl.GL_CULL_FACE)
  # gl.glCullFace(gl.GL_BACK)  # Back-facing polygons will be culled.

  # Rendering.
  program.draw(gl.GL_TRIANGLES, index_buffer)

  # Get the content of the FBO texture.
  rgb = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
  gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, rgb)
  rgb.shape = shape[0], shape[1], 4
  rgb = rgb[::-1, :]
  rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255].

  # Deactivate the frame buffer object.
  fbo.deactivate()

  return rgb


def draw_depth(shape, vertex_buffer, index_buffer, mat_model, mat_view,
               mat_proj):
  """Renders a depth image.

  :param shape: Shape (H, W) of the image to render.
  :param vertex_buffer: Vertex buffer object.
  :param index_buffer: Index buffer object.
  :param mat_model: 4x4 ndarray with the model matrix.
  :param mat_view: 4x4 ndarray with the view matrix.
  :param mat_proj: 4x4 ndarray with the projection matrix.
  :return: HxW ndarray with the rendered depth image.
  """
  # Prepare the OpenGL program.
  program = gloo.Program(_depth_vertex_code, _depth_fragment_code)
  program.bind(vertex_buffer)
  program['u_mv'] = _calc_model_view(mat_model, mat_view)
  program['u_mvp'] = _calc_model_view_proj(mat_model, mat_view, mat_proj)

  # Frame buffer object.
  color_buf = np.zeros(
    (shape[0], shape[1], 4), np.float32).view(gloo.TextureFloat2D)
  depth_buf = np.zeros(
    (shape[0], shape[1]), np.float32).view(gloo.DepthTexture)
  fbo = gloo.FrameBuffer(color=color_buf, depth=depth_buf)
  fbo.activate()

  # OpenGL setup.
  gl.glEnable(gl.GL_DEPTH_TEST)
  gl.glClearColor(0.0, 0.0, 0.0, 0.0)
  gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
  gl.glViewport(0, 0, shape[1], shape[0])

  # Keep the back-face culling disabled because of objects which do not have
  # well-defined surface (e.g. the lamp from the lm dataset).
  gl.glDisable(gl.GL_CULL_FACE)
  # gl.glEnable(gl.GL_CULL_FACE)
  # gl.glCullFace(gl.GL_BACK)  # Back-facing polygons will be culled.

  # Rendering.
  program.draw(gl.GL_TRIANGLES, index_buffer)

  # Get the content of the FBO texture.
  depth = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
  gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, depth)
  depth.shape = shape[0], shape[1], 4
  depth = depth[::-1, :]
  depth = depth[:, :, 0]  # Depth is saved in the first channel

  # Deactivate the frame buffer object.
  fbo.deactivate()

  return depth


def render(model, im_size, K, R, t, clip_near=100, clip_far=2000,
           texture=None, surf_color=None, bg_color=(0.0, 0.0, 0.0, 0.0),
           ambient_weight=0.5, shading='flat', mode='rgb+depth'):
  """Python-based rendering.

  :param model: Dictionary representing a 3D model (see load_ply in inout.py).
  :param im_size: Size (W, H) of the rendered image.
  :param K: 3x3 ndarray with the camera matrix.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :param clip_near: Depth of the near clipping plane.
  :param clip_far: Depth of the far clipping plane.
  :param texture: MxNx3 ndarray with the object model texture.
  :param surf_color: Surface color (R, G, B) used if texture is not defined.
  :param bg_color: Color of the background (R, G, B, A).
  :param ambient_weight: Weight of the ambient light.
  :param shading: Type of shading.
  :param mode: Rendering mode. Options: 'rgb+depth' (default), 'rgb', 'depth'.
  :return: Rendered RGB/depth images (according to the rendering mode).
  """
  # Set texture/color of vertices.
  if texture is not None:
    if texture.max() > 1.0:
      texture = texture.astype(np.float32) / 255.0
    texture = np.flipud(texture)
    texture_uv = model['texture_uv']
    colors = np.zeros((model['pts'].shape[0], 3), np.float32)
  else:
    texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
    if surf_color is None:
      if 'colors' in model.keys():
        assert (model['pts'].shape[0] == model['colors'].shape[0])
        colors = model['colors']
        if colors.max() > 1.0:
          colors /= 255.0  # Color values are expected in range [0, 1].
      else:
        colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5
    else:
      colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])

  # Set the vertex data.
  if mode == 'depth':
    vertices_type = [('a_position', np.float32, 3),
                     ('a_color', np.float32, colors.shape[1])]
    vertices = np.array(list(zip(model['pts'], colors)), vertices_type)
  else:
    if shading == 'flat':
      vertices_type = [('a_position', np.float32, 3),
                       ('a_color', np.float32, colors.shape[1]),
                       ('a_texcoord', np.float32, 2)]
      vertices = np.array(list(zip(model['pts'], colors, texture_uv)),
                          vertices_type)
    elif shading == 'phong':
      vertices_type = [('a_position', np.float32, 3),
                       ('a_normal', np.float32, 3),
                       ('a_color', np.float32, colors.shape[1]),
                       ('a_texcoord', np.float32, 2)]
      vertices = np.array(list(zip(model['pts'], model['normals'],
                                   colors, texture_uv)), vertices_type)
    else:
      raise ValueError('Unknown shading type.')

  # Rendering.
  render_rgb = mode in ['rgb', 'rgb+depth']
  render_depth = mode in ['depth', 'rgb+depth']

  # Model matrix (from object space to world space).
  mat_model = np.eye(4, dtype=np.float32)

  # View matrix (from world space to eye space; transforms also the coordinate
  # system from OpenCV to OpenGL camera space).
  mat_view = np.eye(4, dtype=np.float32)
  mat_view[:3, :3], mat_view[:3, 3] = R, t.squeeze()
  yz_flip = np.eye(4, dtype=np.float32)
  yz_flip[1, 1], yz_flip[2, 2] = -1, -1
  mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system.
  mat_view = mat_view.T  # OpenGL expects column-wise matrix format.

  # Projection matrix.
  mat_proj = _calc_calib_proj(
    K, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

  # Create buffers.
  vertex_buffer = vertices.view(gloo.VertexBuffer)
  index_buffer = \
    model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)

  # Create window.
  # config = app.configuration.Configuration()
  # Number of samples used around the current pixel for multisample
  # anti-aliasing (max is 8).
  # config.samples = 8
  # config.profile = "core"
  # window = app.Window(config=config, visible=False)
  window = app.Window(visible=False)

  global rgb, depth
  rgb = None
  depth = None

  @window.event
  def on_draw(dt):
    window.clear()
    shape = (im_size[1], im_size[0])

    # Render the RGB image.
    if render_rgb:
      global rgb
      rgb = draw_color(
        shape, vertex_buffer, index_buffer, texture, mat_model, mat_view,
        mat_proj, ambient_weight, bg_color, shading)

    # Render the depth image.
    if render_depth:
      global depth
      depth = draw_depth(
        shape, vertex_buffer, index_buffer, mat_model, mat_view, mat_proj)

  # The on_draw function is called framecount+1 times.
  app.run(framecount=0)
  window.close()

  # Set output.
  if mode == 'rgb':
    return rgb
  elif mode == 'depth':
    return depth
  elif mode == 'rgb+depth':
    return rgb, depth
  else:
    raise ValueError('Unknown rendering mode.')


class RendererPython(renderer.Renderer):
  """An interface to the Python based renderer."""

  def __init__(self, width, height):
    """See base class."""
    super(RendererPython, self).__init__(width, height)
    self.models = {}
    self.model_textures = {}
    self.rgb_ims = {}
    self.depth_ims = {}

  def add_object(self, obj_id, model_path):
    """See base class."""
    self.models[obj_id] = inout.load_ply(model_path)

    # Load model texture
    if 'texture_file' in self.models[obj_id].keys():
      model_texture_path = os.path.join(
        os.path.dirname(model_path), self.models[obj_id]['texture_file'])
      model_texture = inout.load_im(model_texture_path)
    else:
      model_texture = None
    self.model_textures[obj_id] = model_texture

  def remove_object(self, obj_id):
    """See base class."""
    del self.models[obj_id]

  def render_object(self, obj_id, R, t, fx, fy, cx, cy, shading='flat',
                    mode='rgb+depth'):
    """See base class."""
    im_size = (self.width, self.height)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    R_a = np.reshape(R, (3, 3))
    t_a = np.reshape(t, (3, 1))

    ren_out = render(
      self.models[obj_id], im_size, K, R_a, t_a,
      ambient_weight=self.light_ambient_weight, shading=shading, mode=mode,
      texture=self.model_textures[obj_id])

    if mode == 'rgb':
      self.rgb_ims[obj_id], self.depth_ims[obj_id] = ren_out, None
    elif mode == 'depth':
      self.rgb_ims[obj_id], self.depth_ims[obj_id] = None, ren_out
    elif mode == 'rgb+depth':
      self.rgb_ims[obj_id], self.depth_ims[obj_id] = ren_out

  def get_color_image(self, obj_id):
    """See base class."""
    assert (obj_id in self.rgb_ims.keys())
    return self.rgb_ims[obj_id]

  def get_depth_image(self, obj_id):
    """See base class."""
    assert (obj_id in self.depth_ims.keys())
    return self.depth_ims[obj_id]
