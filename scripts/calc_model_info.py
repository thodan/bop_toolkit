# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import dataset_params as dataset_p
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

def model(par):
  dataset_params = dataset_p.DatasetParams(par)
  # PARAMETERS.
  ################################################################################
  p = {
    # See dataset_params.py for options.
    'dataset': par.name,

    # Type of input object models.
    'model_type': None,

    # Folder containing the BOP datasets.
    'datasets_path': par.dataset_path,
  }
  ################################################################################


  # Load dataset parameters.
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], p['dataset'], p['model_type'])

  models_info = {}
  for obj_id in dp_model['obj_ids']:
      misc.log('Processing model of object {}...'.format(obj_id))

      model = inout.load_ply(dp_model['model_tpath'].format(obj_id=obj_id))

      # Calculate 3D bounding box.
      ref_pt = model['pts'].min(axis=0).flatten()
      size = (model['pts'].max(axis=0) - ref_pt).flatten()

      # Calculated diameter.
      diameter = misc.calc_pts_diameter(model['pts'])

      models_info[obj_id] = {
          'min_x': ref_pt[0], 'min_y': ref_pt[1], 'min_z': ref_pt[2],
          'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
          'diameter': diameter
      }

  # Save the calculated info about the object models.
  inout.save_json(dp_model['models_info_path'], models_info)
