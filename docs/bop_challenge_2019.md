# BOP Challenge 2019

## The Challenge Task

The methods are evaluated on the task of 6D localization of a varying number of
instances of a varying number of objects (the ViVo task) from a single RGB-D
image.

**Training input:** At training time, a method *M* for 6D object pose estimation
learns from a training set *T = {T<sub>o</sub>}*, where *o* is an object
identifier. Training data *T<sub>o</sub>* may have different forms -- a 3D mesh
model of the object or a set of RGB-D images (synthetic or real) showing object
instances in known 6D poses.

**Test input:** At test time, the method *M* is provided with a list of *test
targets*, each defined by a pair *(I, L)*, where *I* is an image and *L* is a
list *[(o<sub>1</sub>, n<sub>1</sub>), ..., (o<sub>m</sub>, n<sub>m</sub>)]*,
where *n<sub>i</sub>* is the number of instances (at least 1) of object
*o<sub>i</sub>* present in image *I*.

**Test output:** The method *M* produces a list
*E = [E<sub>1</sub>, ..., E<sub>m</sub>]*, where
*E<sub>i</sub>* is a list of *n<sub>i</sub>* pose estimates for object
*o<sub>i</sub>*, each given by a triplet *(R, t, s)* with *R* being a 3x3
rotation matrix, *t* a 3x1 rotation vector and *s* a confidence score.

The ViVo task is equivalent to the *6D localization problem* defined in [2].
In the BOP paper [1], methods were evaluated on a different task --
6D localization of a single instance of a single object (the SiSo task).
The simpler SiSo task was chosen because it allowed to evaluate all relevant
methods out of the box. However, the state of the art has advanced since then
and we are moving to the more challenging ViVo task in this challenge.

## Datasets

All datasets are available in the
[BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
and contain 3D object models and training and test RGB-D images. The training
images show individual objects from different viewpoints and were either
captured by a Kinect-like sensor or obtained by rendering of the 3D object
models. The test images were captured in scenes with graded complexity, often
with clutter and occlusion. For more information, see files *dataset_info.md*
provided with the datasets which can be downloaded from the
[BOP evaluation system](http://bop.felk.cvut.cz/datasets/).

The core datasets:

* **LM-O** (Linemod-Occluded)
* **T-LESS**
* **TUD-L** (TU Dresden Light)
* **IC-BIN** (Doumanoglou et al.)
* **HB** (HomebrewedDB)
* **ITODD** (MVTec ITODD)
* **YCB-V** (YCB-Video)

Other datasets:

* **LM** (Linemod)
* **RU-APC** (Rutgers APC)
* **IC-MI** (Tejani et al.)
* **TYO-L** (Toyota Light)
* **DLR**

## Performance Score

To encourage RGB-only methods, we introduce a modified version of the Visible
Surface Discrepancy which is more tolerant towards misalignment in the Z axis.

## Awards

1. **The Best Method** (prize money: X EUR) - The best-performing method on the 7 core datasets.
2. **The Best Open Source Method** (prize money: X EUR) - The best-performing method on the 7 core
datasets whose source code is publicly available.
3. **The Best Method on Dataset D** (prize money: Y EUR) - D can be any of the 12 available datasets.

## How to Participate

### Challenge Rules

1. For training, you can use the provided object models and training images
(both real and rendered). You can also render extra training images using the
object models.
2. Not a single pixel of test images may be used in training, nor the ground
truth poses provided for the test images. The distribution of object poses in
the test scenes (from *bop_toolkit/dataset_params.py*) is the only information
about the test set that can be used in training.
3. To become the best method on the 7 core datasets (including the best open
source method), parameters of the method need to be constant across all objects
and datasets.
4. If you want your results to be included in a publication about the challenge,
a documentation of results (provided through the online submission form) is
required. Without the documentation, your results will be listed on the website
but not included in the publication.

### Submission of Results

To have your method evaluated, run it on the ViVo task and submit the results in
the format described below to the
[BOP evaluation system](http://bop.felk.cvut.cz).

Results for all test images from one dataset are saved in one CSV file, with one
pose estimate per line in the following format:

```
scene_id,im_id,obj_id,score,R,t,time
```

where:
* a triplet *(scene_id, im_id, obj_id)* defines a test target [1].
* *score* is a confidence of the estimate (the range of values is not
restricted).
* *R* is a 3x3 rotation matrix whose elements are saved row-wise and separated
by a white space (i.e. ```r11 r12 r13 r21 r22 r23 r31 r32 r33```, where *rij* is
an element in the *i*-th row and *j*-th column of the matrix).
* *t* is a 3x1 translation vector (in mm) whose elements are separated by a
white space (i.e. ```t1 t2 t3```).
* *time* ...

The method is expected to produce a single pose estimate per test target.
If multiple pose estimates are provided, the one with the highest score is used.

Example results can be found
[here](http://ptak.felk.cvut.cz/6DB/public/bop_sample_results).

P = K * [R t] is the camera matrix that transforms 3D point p\_m = [x, y, z, 1]'
in the model coordinate system to 2D point p\_i = [u, v, 1]' in the image
coordinate system: s * p\_i = P * p\_m. The camera coordinate system is defined
as in OpenCV with the camera looking along the Z axis. K is provided with the
test images and might be different for each image.

* TIME is the wall time from the point right after the raw data (images, 3D
object models etc.) is loaded to the point when the final pose estimates are
available (a single real number in seconds, -1 if not available).
* SCORE is a confidence of the estimate (the range of values is not

The list of objects that are present in an image can be obtained from the file
*gt.yml* provided for each test scene.

## Organizers

...

## References

[1] Hodan, Michel et al. "BOP: Benchmark for 6D Object Pose Estimation" ECCV'18.
