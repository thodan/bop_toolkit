# Unit tests of the BOP toolkit

## Test of eval_bop19_pose_test.py
```
python bop_toolkit_lib/tests/eval_bop19_pose_test.py
```

## Test of eval_bop19_pose.py with torch implementation
```
python bop_toolkit_lib/tests/eval_bop19_pose_test_gpu.py

# add more false positive samples
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --num_false_positives 10000 --use_torch
```

## Test of eval_bop24_pose.py
```
python bop_toolkit_lib/tests/eval_bop24_pose_test.py

# add more false positive samples
python bop_toolkit_lib/tests/eval_bop24_pose_test.py --num_false_positives 10000
```

Results:
<p align="center">
  <img src=./unit_test_results.png width="100%"/>
</p>