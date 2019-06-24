# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""I/O functions."""

import os
import struct
import numpy as np
import imageio
import png
import ruamel.yaml as yaml

from bop_toolkit import misc


def load_yaml(path):
  """Loads content of a YAML file.

  :param path: Path to the YAML file.
  :return: Content of the loaded YAML file.
  """
  with open(path, 'r') as f:
    content = yaml.load(f, Loader=yaml.CLoader)
  return content


def save_yaml(path, content):
  """Saves the provided content to a YAML file.

  :param path: Path to the output YAML file.
  :param content: Dictionary/list to save.
  """
  with open(path, 'w') as f:
    yaml.dump(content, f, Dumper=yaml.CDumper, width=100000)


def load_cam_params(path):
  """Loads camera parameters from a YAML file.

  :param path: Path to the YAML file.
  :return: Dictionary with the following items:
   - 'im_size': (width, height).
   - 'K': 3x3 camera matrix.
   - 'depth_scale': Scale factor to convert the depth images to mm (optional).
  """
  with open(path, 'r') as f:
    c = yaml.load(f, Loader=yaml.CLoader)

  cam = {
    'im_size': (c['width'], c['height']),
    'K': np.array([[c['fx'], 0.0, c['cx']],
                   [0.0, c['fy'], c['cy']],
                   [0.0, 0.0, 1.0]])
  }

  if 'depth_scale' in c.keys():
    cam['depth_scale'] = float(c['depth_scale'])

  return cam


def load_im(path):
  """Loads an image from a file.

  :param path: Path to the image file to load.
  :return: ndarray with the loaded image.
  """
  im = imageio.imread(path)
  return im


def save_im(path, im, jpg_quality=95):
  """Saves an image to a file.

  :param path: Path to the output image file.
  :param im: ndarray with the image to save.
  :param jpg_quality: Quality of the saved image (applies only to JPEG).
  """
  ext = os.path.splitext(path)[1][1:]
  if ext.lower() in ['jpg', 'jpeg']:
    imageio.imwrite(path, im, quality=jpg_quality)
  else:
    imageio.imwrite(path, im)


def load_depth(path):
  """Loads a depth image from a file.

  :param path: Path to the depth image file to load.
  :return: ndarray with the loaded depth image.
  """
  d = imageio.imread(path)
  return d.astype(np.float32)


def save_depth(path, im):
  """Saves a depth image (16-bit) to a PNG file.

  :param path: Path to the output depth image file.
  :param im: ndarray with the depth image to save.
  """
  im_uint16 = np.round(im).astype(np.uint16)

  # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
  w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
  with open(path, 'wb') as f:
    w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def load_scene_camera(path):
  """Loads content of a YAML file with information about the scene camera.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the YAML file.
  :return: Dictionary with the loaded content.
  """
  with open(path, 'r') as f:
    info = yaml.load(f, Loader=yaml.CLoader)
    for eid in info.keys():
      if 'cam_K' in info[eid].keys():
        info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
      if 'cam_R_w2c' in info[eid].keys():
        info[eid]['cam_R_w2c'] =\
          np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
      if 'cam_t_w2c' in info[eid].keys():
        info[eid]['cam_t_w2c'] =\
          np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
  return info


def save_scene_camera(path, scene_camera):
  """Saves information about the scene camera to a YAML file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output YAML file.
  :param scene_camera: Dictionary to save to the YAML file.
  """
  for im_id in sorted(scene_camera.keys()):
    im_info = scene_camera[im_id]
    if 'cam_K' in im_info.keys():
      im_info['cam_K'] = im_info['cam_K'].flatten().tolist()
    if 'cam_R_w2c' in im_info.keys():
      im_info['cam_R_w2c'] = im_info['cam_R_w2c'].flatten().tolist()
    if 'cam_t_w2c' in im_info.keys():
      im_info['cam_t_w2c'] = im_info['cam_t_w2c'].flatten().tolist()
  with open(path, 'w') as f:
    yaml.dump(scene_camera, f, Dumper=yaml.CDumper, width=10000)


def load_scene_gt(path):
  """Loads content of a YAML file with ground-truth annotations.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the YAML file.
  :return: Dictionary with the loaded content.
  """
  with open(path, 'r') as f:
    gts = yaml.load(f, Loader=yaml.CLoader)
    for im_id, gts_im in gts.items():
      for gt in gts_im:
        if 'cam_R_m2c' in gt.keys():
          gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
        if 'cam_t_m2c' in gt.keys():
          gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
  return gts


def save_scene_gt(path, scene_gt):
  """Saves ground-truth annotations to a YAML file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output YAML file.
  :param scene_gt: Dictionary to save to the YAML file.
  """
  for im_id in sorted(scene_gt.keys()):
    im_gts = scene_gt[im_id]
    for gt in im_gts:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
      if 'obj_bb' in gt.keys():
        gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
  with open(path, 'w') as f:
    yaml.dump(scene_gt, f, Dumper=yaml.CDumper, width=10000)


def load_bop_results(path, version='bop_challenge_2019'):
  """Loads 6D object pose estimates from a file.

  :param path: Path to a file with pose estimates.
  :return: List of loaded poses.
  """
  results = []

  # See docs/bop_challenge_2019.md for details.
  if version == 'bop_challenge_2019':
    header = 'scene_id,im_id,obj_id,score,R,t,time'
    with open(path, 'r') as f:
      line_id = 0
      for line in f:
        line_id += 1
        if line_id == 1 and header in line:
          continue
        else:
          elems = line.split(',')
          if len(elems) != 7:
            raise ValueError(
              'A line does not have 7 comma-sep. elements: {}'.format(line))
          results.append({
            'scene_id': int(elems[0]),
            'im_id': int(elems[1]),
            'obj_id': int(elems[2]),
            'score': float(elems[3]),
            'R': np.array(map(float, elems[4].split())).reshape((3, 3)),
            't': np.array(map(float, elems[5].split())).reshape((3, 1)),
            'time': float(elems[6])})
  else:
    raise ValueError('Unknown version of BOP results.')

  return results


def save_bop_results(path, results, version='bop_challenge_2019'):
  """Saves 6D object pose estimates to a YAML file.

  :param path: Path to the output YAML file.
  :param res: Dictionary with pose estimates.
  :param run_time: Time which the evaluated method took to make the estimates.
  """
  # See docs/bop_challenge_2019.md for details.
  if version == 'bop_challenge_2019':
    lines = ['scene_id,im_id,obj_id,score,R,t,time']
    for res in results:
      if 'time' in res:
        run_time = res['time']
      else:
        run_time = -1

      lines.append('{scene_id},{im_id},{obj_id},{score},{R},{t},{time}'.format(
        scene_id=res['scene_id'],
        im_id=res['im_id'],
        obj_id=res['obj_id'],
        score=res['score'],
        R=' '.join(map(str, res['R'].flatten().tolist())),
        t=' '.join(map(str, res['t'].flatten().tolist())),
        time=run_time))

    with open(path, 'w') as f:
      f.write('\n'.join(lines))

  else:
    raise ValueError('Unknown version of BOP results.')


def save_errors(path, errors):
  """Saves errors of pose estimates to a YAML file.

  See scripts/eval_calc_errors.py for details.

  :param path: Path to the output YAML file.
  :return: Dictionary with errors to save.
  """
  with open(path, 'w') as f:
    line_tpl = '- {{im_id: {:d}, obj_id: {:d}, est_id: {:d}, ' \
               'score: {:f}, errors: {}}}\n'
    error_tpl = '{:d}: [{}]'
    txt = ''
    for e in errors:
      txt_errors_elems = []
      for gt_id, error in e['errors'].items():
        error_str = map(str, error)
        txt_errors_elems.append(error_tpl.format(gt_id, ', '.join(error_str)))
      txt_errors = '{' + ', '.join(txt_errors_elems) + '}'
      txt += line_tpl.format(e['im_id'], e['obj_id'], e['est_id'],
                             e['score'], txt_errors)
    f.write(txt)


def load_errors(path):
  """Loads errors of pose estimates from a YAML file.

  See scripts/eval_calc_errors.py for details.

  :param path: Path to a YAML file with errors.
  :return: Dictionary with the loaded errors.
  """
  with open(path, 'r') as f:
    errors = yaml.load(f, Loader=yaml.CLoader)
  return errors


def load_ply(path):
  """Loads a 3D mesh model from a PLY file.

  :param path: Path to a PLY file.
  :return: The loaded model given by a dictionary with items:
   - 'pts' (nx3 ndarray)
   - 'normals' (nx3 ndarray), optional
   - 'colors' (nx3 ndarray), optional
   - 'faces' (mx3 ndarray), optional
   - 'texture_uv' (nx2 ndarray), optional
   - 'texture_uv_face' (mx6 ndarray), optional
   - 'texture_file' (string), optional
  """
  f = open(path, 'rb')

  # Only triangular faces are supported.
  face_n_corners = 3

  n_pts = 0
  n_faces = 0
  pt_props = []
  face_props = []
  is_binary = False
  header_vertex_section = False
  header_face_section = False
  texture_file = None

  # Read the header.
  while True:

    # Strip the newline character(s).
    line = f.readline().rstrip('\n').rstrip('\r')

    if line.startswith('comment TextureFile'):
      texture_file = line.split()[-1]
    elif line.startswith('element vertex'):
      n_pts = int(line.split()[-1])
      header_vertex_section = True
      header_face_section = False
    elif line.startswith('element face'):
      n_faces = int(line.split()[-1])
      header_vertex_section = False
      header_face_section = True
    elif line.startswith('element'):  # Some other element.
      header_vertex_section = False
      header_face_section = False
    elif line.startswith('property') and header_vertex_section:
      # (name of the property, data type)
      pt_props.append((line.split()[-1], line.split()[-2]))
    elif line.startswith('property list') and header_face_section:
      elems = line.split()
      if elems[-1] == 'vertex_indices':
        # (name of the property, data type)
        face_props.append(('n_corners', elems[2]))
        for i in range(face_n_corners):
          face_props.append(('ind_' + str(i), elems[3]))
      elif elems[-1] == 'texcoord':
        # (name of the property, data type)
        face_props.append(('texcoord', elems[2]))
        for i in range(face_n_corners * 2):
          face_props.append(('texcoord_ind_' + str(i), elems[3]))
      else:
        misc.log('Warning: Not supported face property: ' + elems[-1])
    elif line.startswith('format'):
      if 'binary' in line:
        is_binary = True
    elif line.startswith('end_header'):
      break

  # Prepare data structures.
  model = {}
  if texture_file is not None:
    model['texture_file'] = texture_file
  model['pts'] = np.zeros((n_pts, 3), np.float)
  if n_faces > 0:
    model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

  pt_props_names = [p[0] for p in pt_props]
  face_props_names = [p[0] for p in face_props]

  is_normal = False
  if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
    is_normal = True
    model['normals'] = np.zeros((n_pts, 3), np.float)

  is_color = False
  if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
    is_color = True
    model['colors'] = np.zeros((n_pts, 3), np.float)

  is_texture_pt = False
  if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
    is_texture_pt = True
    model['texture_uv'] = np.zeros((n_pts, 2), np.float)

  is_texture_face = False
  if {'texcoord'}.issubset(set(face_props_names)):
    is_texture_face = True
    model['texture_uv_face'] = np.zeros((n_faces, 6), np.float)

  # Formats for the binary case.
  formats = {
    'float': ('f', 4),
    'double': ('d', 8),
    'int': ('i', 4),
    'uchar': ('B', 1)
  }

  # Load vertices.
  for pt_id in range(n_pts):
    prop_vals = {}
    load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                  'red', 'green', 'blue', 'texture_u', 'texture_v']
    if is_binary:
      for prop in pt_props:
        format = formats[prop[1]]
        read_data = f.read(format[1])
        val = struct.unpack(format[0], read_data)[0]
        if prop[0] in load_props:
          prop_vals[prop[0]] = val
    else:
      elems = f.readline().rstrip('\n').rstrip('\r').split()
      for prop_id, prop in enumerate(pt_props):
        if prop[0] in load_props:
          prop_vals[prop[0]] = elems[prop_id]

    model['pts'][pt_id, 0] = float(prop_vals['x'])
    model['pts'][pt_id, 1] = float(prop_vals['y'])
    model['pts'][pt_id, 2] = float(prop_vals['z'])

    if is_normal:
      model['normals'][pt_id, 0] = float(prop_vals['nx'])
      model['normals'][pt_id, 1] = float(prop_vals['ny'])
      model['normals'][pt_id, 2] = float(prop_vals['nz'])

    if is_color:
      model['colors'][pt_id, 0] = float(prop_vals['red'])
      model['colors'][pt_id, 1] = float(prop_vals['green'])
      model['colors'][pt_id, 2] = float(prop_vals['blue'])

    if is_texture_pt:
      model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
      model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

  # Load faces.
  for face_id in range(n_faces):
    prop_vals = {}
    if is_binary:
      for prop in face_props:
        format = formats[prop[1]]
        val = struct.unpack(format[0], f.read(format[1]))[0]
        if prop[0] == 'n_corners':
          if val != face_n_corners:
            raise ValueError('Only triangular faces are supported.')
        elif prop[0] == 'texcoord':
          if val != face_n_corners * 2:
            raise ValueError('Wrong number of UV face coordinates.')
        else:
          prop_vals[prop[0]] = val
    else:
      elems = f.readline().rstrip('\n').rstrip('\r').split()
      for prop_id, prop in enumerate(face_props):
        if prop[0] == 'n_corners':
          if int(elems[prop_id]) != face_n_corners:
            raise ValueError('Only triangular faces are supported.')
        elif prop[0] == 'texcoord':
          if int(elems[prop_id]) != face_n_corners * 2:
            raise ValueError('Wrong number of UV face coordinates.')
        else:
          prop_vals[prop[0]] = elems[prop_id]

    model['faces'][face_id, 0] = int(prop_vals['ind_0'])
    model['faces'][face_id, 1] = int(prop_vals['ind_1'])
    model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    if is_texture_face:
      for i in range(6):
        model['texture_uv_face'][face_id, i] = float(
          prop_vals['texcoord_ind_{}'.format(i)])

  f.close()

  return model


def save_ply(path, model, extra_header_comments=None):
  """Saves a 3D mesh model to a PLY file.

  :param path: Path to a PLY file.
  :param model: 3D model given by a dictionary with items:
   - 'pts' (nx3 ndarray)
   - 'normals' (nx3 ndarray, optional)
   - 'colors' (nx3 ndarray, optional)
   - 'faces' (mx3 ndarray, optional)
   - 'texture_uv' (nx2 ndarray, optional)
   - 'texture_uv_face' (mx6 ndarray, optional)
   - 'texture_file' (string, optional)
  :param extra_header_comments: Extra header comment (optional).
  """
  pts = model['pts']
  pts_colors = model['colors'] if 'colors' in model.keys() else np.array([])
  pts_normals = model['normals'] if 'normals' in model.keys() else np.array([])
  faces = model['faces'] if 'faces' in model.keys() else np.array([])
  texture_uv = model[
    'texture_uv'] if 'texture_uv' in model.keys() else np.array([])
  texture_uv_face = model[
    'texture_uv_face'] if 'texture_uv_face' in model.keys() else np.array([])
  texture_file = model[
    'texture_file'] if 'texture_file' in model.keys() else np.array([])

  save_ply2(path, pts, pts_colors, pts_normals, faces, texture_uv,
            texture_uv_face,
            texture_file, extra_header_comments)


def save_ply2(path, pts, pts_colors=None, pts_normals=None, faces=None,
              texture_uv=None, texture_uv_face=None, texture_file=None,
              extra_header_comments=None):
  """Saves a 3D mesh model to a PLY file.

  :param path: Path to the resulting PLY file.
  :param pts: nx3 ndarray with vertices.
  :param pts_colors: nx3 ndarray with vertex colors (optional).
  :param pts_normals: nx3 ndarray with vertex normals (optional).
  :param faces: mx3 ndarray with mesh faces (optional).
  :param texture_uv: nx2 ndarray with per-vertex UV texture coordinates
    (optional).
  :param texture_uv_face: mx6 ndarray with per-face UV texture coordinates
    (optional).
  :param texture_file: Path to a texture image -- relative to the resulting
    PLY file (optional).
  :param extra_header_comments: Extra header comment (optional).
  """
  pts_colors = np.array(pts_colors)
  if pts_colors is not None:
    assert (len(pts) == len(pts_colors))

  valid_pts_count = 0
  for pt_id, pt in enumerate(pts):
    if not np.isnan(np.sum(pt)):
      valid_pts_count += 1

  f = open(path, 'w')
  f.write(
    'ply\n'
    'format ascii 1.0\n'
    # 'format binary_little_endian 1.0\n'
  )

  if texture_file is not None:
    f.write('comment TextureFile {}\n'.format(texture_file))

  if extra_header_comments is not None:
    for comment in extra_header_comments:
      f.write('comment {}\n'.format(comment))

  f.write(
    'element vertex ' + str(valid_pts_count) + '\n'
    'property float x\n'
    'property float y\n'
    'property float z\n'
  )
  if pts_normals is not None:
    f.write(
      'property float nx\n'
      'property float ny\n'
      'property float nz\n'
    )
  if pts_colors is not None:
    f.write(
      'property uchar red\n'
      'property uchar green\n'
      'property uchar blue\n'
    )
  if texture_uv is not None:
    f.write(
      'property float texture_u\n'
      'property float texture_v\n'
    )
  if faces is not None:
    f.write(
      'element face ' + str(len(faces)) + '\n'
      'property list uchar int vertex_indices\n'
    )
  if texture_uv_face is not None:
    f.write(
      'property list uchar float texcoord\n'
    )
  f.write('end_header\n')

  format_float = "{:.4f}"
  format_2float = " ".join((format_float for _ in range(2)))
  format_3float = " ".join((format_float for _ in range(3)))
  format_int = "{:d}"
  format_3int = " ".join((format_int for _ in range(3)))

  # Save vertices.
  for pt_id, pt in enumerate(pts):
    if not np.isnan(np.sum(pt)):
      f.write(format_3float.format(*pts[pt_id].astype(float)))
      if pts_normals is not None:
        f.write(' ')
        f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
      if pts_colors is not None:
        f.write(' ')
        f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
      if texture_uv is not None:
        f.write(' ')
        f.write(format_2float.format(*texture_uv[pt_id].astype(float)))
      f.write('\n')

  # Save faces.
  if faces is not None:
    for face_id, face in enumerate(faces):
      line = ' '.join(map(str, map(int, [len(face)] + list(face.squeeze()))))
      if texture_uv_face is not None:
        uv = texture_uv_face[face_id]
        line += ' ' + ' '.join(
          map(str, [len(uv)] + map(float, list(uv.squeeze()))))
      f.write(line)
      f.write('\n')

  f.close()
