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

Results:
<p align="center">
  <img src=./run_time_localization_tasks.png width="100%"/>
</p>

<p align="center">
  <img src=./run_time_detection_tasks.png width="80%"/>
</p>