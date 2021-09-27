import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../cocoapi/PythonAPI'))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from bop_toolkit_lib import pycoco_utils
import skimage.io as io
import argparse
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
  # Minimum visible surface fraction of a valid GT pose.
  # -1 == k most visible GT poses will be considered, where k is given by
  # the "inst_count" item loaded from "targets_filename".
  'visib_gt_min': -1,

  # Names of files with detection results for which to calculate the Average Precisions
  # (assumed to be stored in folder p['results_path']). 
  'result_filenames': [
    'json/file/with/coco/results',
  ],

  # Folder with results to be evaluated.
  'results_path': config.results_path,

  # Folder for the calculated pose errors and performance scores.
  'eval_path': config.eval_path,
  
  # Folder with BOP datasets.
  'datasets_path': config.datasets_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  'targets_filename': 'test_targets_bop19.json',
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--result_filenames',
                    default=','.join(p['result_filenames']),
                    help='Comma-separated names of files with results.')
parser.add_argument('--results_path', default=p['results_path'])
parser.add_argument('--eval_path', default=p['eval_path'])
parser.add_argument('--targets_filename', default=p['targets_filename'])
args = parser.parse_args()

p['result_filenames'] = args.result_filenames.split(',')
p['results_path'] = str(args.results_path)
p['eval_path'] = str(args.eval_path)
p['targets_filename'] = str(args.targets_filename)


# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p['result_filenames']:

  misc.log('===========')
  misc.log('EVALUATING: {}'.format(result_filename))
  misc.log('===========')
  
  # Parse info about the method and the dataset from the filename.
  result_name = os.path.splitext(os.path.basename(result_filename))[0]
  result_info = result_name.split('_')
  method = str(result_info[0])
  dataset_info = result_info[1].split('-')
  dataset = str(dataset_info[0])
  split = str(dataset_info[1])
  split_type = str(dataset_info[2]) if len(dataset_info) > 2 else None

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'eval'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)
  
  # Load coco results
  coco_results = inout.load_json(os.path.join(p['results_path'], result_filename), keys_to_int=True)
  
  # Merge coco scene annotations and results 
  for i, scene_id in enumerate(dp_split['scene_ids']):
    
    scene_coco_ann_path = dp_split['scene_gt_coco_tpath'].format(scene_id=scene_id)
    scene_coco_ann = inout.load_json(scene_coco_ann_path, keys_to_int=True)
    
    scene_coco_results = coco_results[scene_id] if scene_id in coco_results else []
    
    if i == 0:
      dataset_coco_ann = scene_coco_ann
      dataset_coco_results = scene_coco_results
    else:
      dataset_coco_ann, image_id_offset = pycoco_utils.merge_coco_annotations(dataset_coco_ann, scene_coco_ann)
      dataset_coco_results = pycoco_utils.merge_coco_results(dataset_coco_results, scene_coco_results, image_id_offset)


annType = 'segm' #'bbox'      #specify type here

#initialize COCO ground truth api
cocoGt=COCO(dataset_coco_ann)
cocoDt=cocoGt.loadRes(dataset_coco_results)

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
