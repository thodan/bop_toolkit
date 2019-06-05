# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Parameters of the BOP datasets."""

import math
from os.path import join

from bop_toolkit import inout


def get_dataset_params(
      datasets_path, dataset_name, model_type=None, train_type=None,
      val_type=None, test_type=None, cam_type=None):
  """Return parameters (camera parameters, paths etc.) for a specified dataset.

  :param datasets_path: Path to a folder with datasets.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :param model_type: Type of object models.
  :param train_type: Type of training images.
  :param val_type: Type of validation images.
  :param test_type: Type of test images.
  :param cam_type: Type of camera.
  :return: Dictionary with parameters for the specified dataset.
  """

  p = {
    'name': dataset_name,
    'model_type': model_type,
    'train_type': train_type,
    'val_type': val_type,
    'test_type': test_type,
    'cam_type': cam_type,
    'base_path': join(datasets_path, dataset_name),
    'cam_params_path': join(datasets_path, dataset_name, 'camera.yml')
  }

  # Dataset-specific parameters.
  # ----------------------------------------------------------------------------
  # Linemod (LM).
  if dataset_name == 'lm':
    p['obj_ids'] = range(1, 16)
    p['scene_ids'] = range(1, 16)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (600.90, 1102.35)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (0, 0.5 * math.pi)

  # Linemod-Occluded (LM).
  elif dataset_name == 'lmo':
    p['obj_ids'] = [1, 5, 6, 8, 9, 10, 11, 12]
    p['scene_ids'] = [2]
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (346.31, 1499.84)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (0, 0.5 * math.pi)

  # T-LESS.
  elif dataset_name == 'tless':
    p['obj_ids'] = range(1, 31)
    p['scene_ids'] = range(1, 21)

    # Default types of object models, training and test data, and camera.
    if p['model_type'] is None:
      p['model_type'] = 'cad'
    if p['train_type'] is None:
      p['train_type'] = 'primesense'
    if p['test_type'] is None:
      p['test_type'] = 'primesense'
    if p['cam_type'] is None:
      p['cam_type'] = 'primesense'

    # T-LESS includes images captured by three sensors.
    cam_filename = 'camera_{}.yml'.format(p['cam_type'])
    p['cam_params_path'] = join(p['base_path'], cam_filename)

    if p['test_type'] in ['primesense', 'kinect']:
      p['test_im_size'] = (720, 540)
    elif p['test_type'] == 'canon':
      p['test_im_size'] = (2560, 1920)

    if p['train_type'] in ['primesense', 'kinect']:
      p['train_im_size'] = (400, 400)
    elif p['train_type'] == 'canon':
      p['train_im_size'] = (1900, 1900)
    elif p['train_type'] == 'render_reconst':
      p['train_im_size'] = (1280, 1024)

    # The following holds for Primesense, but is similar for the other sensors.
    p['test_depth_range'] = (649.89, 940.04)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (-0.5 * math.pi, 0.5 * math.pi)

  # TU Dresden Light (TUD-L).
  elif dataset_name == 'tudl':
    p['obj_ids'] = range(1, 4)
    p['scene_ids'] = range(1, 4)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (851.29, 2016.14)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (-0.4363, 0.5 * math.pi)  # (-25, 90) [deg].

  # Toyota Light (TYO-L).
  elif dataset_name == 'tyol':
    p['obj_ids'] = range(1, 22)
    p['scene_ids'] = range(1, 22)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    # Not calculated yet.
    p['test_depth_range'] = None
    p['test_azimuth_range'] = None
    p['test_elev_range'] = None

  # Rutgers APC (RU-APC).
  elif dataset_name == 'ruapc':
    p['obj_ids'] = range(1, 15)
    p['scene_ids'] = range(1, 15)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (594.41, 739.12)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (-0.5 * math.pi, 0.5 * math.pi)

  # Tejani et al. (IC-MI).
  elif dataset_name == 'icmi':
    p['obj_ids'] = range(1, 7)
    p['scene_ids'] = range(1, 7)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (509.12, 1120.41)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (0, 0.5 * math.pi)

  # Doumanoglou et al. (IC-BIN).
  elif dataset_name == 'icbin':
    p['obj_ids'] = range(1, 3)
    p['scene_ids'] = range(1, 4)
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)

    p['test_depth_range'] = (454.56, 1076.29)
    p['test_azimuth_range'] = (0, 2 * math.pi)
    p['test_elev_range'] = (-1.0297, 0.5 * math.pi)  # (-59, 90) [deg].

  else:
    raise ValueError('Unknown BOP dataset.')

  # Camera parameters.
  # ----------------------------------------------------------------------------
  p['cam'] = inout.load_cam_params(p['cam_params_path'])

  # Generic parameters.
  # ----------------------------------------------------------------------------
  def get_data_base_path(data_name, data_type):
    data_path = join(p['base_path'], data_name)
    if data_type is not None:
      data_path += '_' + data_type
    return data_path

  # Paths to models, and to training, validation and test data.
  models_path = get_data_base_path('models', p['model_type'])
  train_path = get_data_base_path('train', p['train_type'])
  val_path = get_data_base_path('val', p['val_type'])
  test_path = get_data_base_path('test', p['test_type'])

  def get_data_path_templates(base_path, prefix):
    return {
      prefix + '_info_tpath': join(base_path, '{scene_id:06d}', 'info.yml'),
      prefix + '_gt_tpath': join(base_path, '{scene_id:06d}', 'gt.yml'),
      prefix + '_gt_stats_tpath': join(base_path + '_gt_stats', '{scene_id:06d}_delta={delta:d}.yml'),
      prefix + '_rgb_tpath': join(base_path, '{scene_id:06d}', 'rgb', '{im_id:06d}.png'),
      prefix + '_depth_tpath': join(base_path, '{scene_id:06d}', 'depth', '{im_id:06d}.png'),
      prefix + '_mask_tpath': join(base_path, '{scene_id:06d}', 'mask', '{im_id:06d}_{gt_id:06d}.png'),
      prefix + '_mask_visib_tpath': join(base_path, '{scene_id:06d}', 'mask_visib', '{im_id:06d}_{gt_id:06d}.png'),
    }

  # Training/validation/test data path templates.
  p.update(get_data_path_templates(train_path, 'train'))
  p.update(get_data_path_templates(val_path, 'val'))
  p.update(get_data_path_templates(test_path, 'test'))

  # Path templates for object models.
  p['model_tpath'] = join(models_path, 'obj_{obj_id:06d}.ply')
  p['models_info_path'] = join(models_path, 'models_info.yml')

  return p
