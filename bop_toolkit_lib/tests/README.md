# Testing the evaluation scripts
The tests verify consistency of AR/AP score between the evaluation scripts output and official bop website scores for a few chosen submissions.   
## 2D detection tasks
```
python bop_toolkit_lib/tests/eval_bop22_coco_test.py
```
If an error occurs due to missing `scene_gt_coco.json`, run:   
```python scripts/calc_gt_coco.py --dataset <dataset_name>```

## 6D localization task with CPU/GPU implementation
```
python bop_toolkit_lib/tests/eval_bop19_pose_test.py
python bop_toolkit_lib/tests/eval_bop19_pose_test.py --use_gpu
```

## 6D detection task with CPU/GPU implementation
```
python bop_toolkit_lib/tests/eval_bop24_pose_test.py
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --use_gpu

# add more false positive samples
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --num_false_positives 10000
```

## Testing against ground-truth
First generate all ground-truth result files:  
```
sh bop_toolkit_lib/tests/create_all_gt_files.sh
```

Run eval test scripts with `--gt_from_datasets` argument, e.g.:

```
python bop_toolkit_lib/tests/eval_bop19_pose_test.py --gt_from_datasets lmo,ycbv,tless,itodd,hb,icbin,tudl
python bop_toolkit_lib/tests/eval_bop22_coco_test.py --gt_from_datasets lmo,ycbv,tless,itodd,hb,icbin,tudl,hopev2,hot3d,handal,ipd,xyzibd,itoddmv
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --gt_from_datasets hopev2,hot3d,handal,ipd,xyzibd,itoddmv
```

## Run time results:
<p align="center">
  <img src=./run_time_localization_tasks.png width="100%"/>
</p>

<p align="center">
  <img src=./run_time_detection_tasks.png width="80%"/>
</p>