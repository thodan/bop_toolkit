# Unit tests of the BOP toolkit

## Test of 6D localization task with CPU implementation
```
python bop_toolkit_lib/tests/eval_bop19_pose_test.py
```

## Test of 6D localization task with GPU implementation
```
python bop_toolkit_lib/tests/eval_bop19_pose_test_gpu.py
```

## Test of 6D detection task with CPU implementation
```
python bop_toolkit_lib/tests/eval_bop24_pose_test.py

# add more false positive samples
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --num_false_positives 10000
```

## Test of 6D detection task with GPU implementation
```
python bop_toolkit_lib/tests/eval_bop24_pose_test_gpu.py

# add more false positive samples
python bop_toolkit_lib/tests/eval_bop24_pose_test_gpu.py --num_false_positives 10000
```

## Test of 6D detetection with only objects visible > 10% in the image
```
python scripts/eval_bop24_pose.py --renderer_type=vispy --results_path ./bop_toolkit_lib/tests/data/ --eval_path ./bop_toolkit_lib/tests/data/ --result_filenames unittest-minVisib0_tless-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv --num_worker 10
```

Results:
<p align="center">
  <img src=./run_time_localization_tasks.png width="100%"/>
</p>

<p align="center">
  <img src=./run_time_detection_tasks.png width="80%"/>
</p>