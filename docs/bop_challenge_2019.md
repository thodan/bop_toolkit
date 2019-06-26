# BOP Challenge 2019

## The Challenge Task

The challenge is on the task of 6D localization of a *varying* number of
instances of a *varying* number of objects in a single RGB-D image (the ViVo
task).

**Training input:** At training time, method *M* learns using training set
*T = {T<sub>o</sub>}*, where *o* is an object identifier. Training data
*T<sub>o</sub>* may have different forms -- a 3D mesh model of the object or a
set of RGB-D images (synthetic or real) showing object instances in known 6D
poses.

**Test input:** At test time, method *M* is provided with a list of *test
targets*, each defined by a pair *(I, L)*, where *I* is an image and *L* is a
list *[(o<sub>1</sub>, n<sub>1</sub>), ..., (o<sub>m</sub>, n<sub>m</sub>)]*,
where *n<sub>i</sub>* is the number of instances of object
*o<sub>i</sub>* present in image *I*.

**Test output:** Method *M* produces a list
*E = [E<sub>1</sub>, ..., E<sub>m</sub>]*, where *E<sub>i</sub>* is a list of
*n<sub>i</sub>* pose estimates for instances of object *o<sub>i</sub>*. Each
estimate is given by a triplet *(R, t, s)* with *R* being a 3x3 rotation matrix,
*t* a 3x1 rotation vector and *s* a confidence score.

The ViVo task is referred to as the *6D localization problem* in [2]. In the BOP
paper [1], methods were evaluated on a different task -- 6D localization of a
single instance of a single object (the SiSo task). The simpler SiSo task was
chosen because it allowed to evaluate all relevant methods out of the box.
However, the state of the art has advanced since then and we are moving to the
more challenging ViVo task in this challenge.

## Datasets

**Content of the datasets:** The datasets include 3D object models and training
and test RGB-D images annotated with ground-truth 6D object poses and intrinsic
parameters of the camera. The 3D object models were created manually or using
KinectFusion-like systems for 3D surface reconstruction [3].

**Training and test images:** The training images show individual objects from
different viewpoints and are either captured by an RGB-D/Gray-D sensor or
obtained by rendering of the 3D object models. The test images were captured in
scenes with graded complexity, often with clutter and occlusion.

**Rules for training:** For training, method *M* can use the provided object
models and training images, but can also render extra training images using the
object models. Not a single pixel of test images may be used in training, nor
the ground truth poses provided for the test images. The distribution of object
poses in the test images (provided in *bop_toolkit/dataset_params.py*) is the
only information about the test set that can be used in training.

**Challenge datasets:** The following datasets have been converted to the
[BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
and can be downloaded from the
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

## Awards

1. **The Best Method** (prize money: X EUR) - The best-performing method on
the 7 core datasets.
2. **The Best RGB-Only Method** (prize money: X EUR) - The best-performing
RGB-only method on the 7 core datasets.
3. **The Best Open Source Method** (prize money: Y EUR) - The best-performing
method on the 7 core datasets whose source code is publicly available.
4. **The Best Method on Dataset D** - D can be any of the 12 available datasets.

## Evaluation Methodology

The error of a 6D object pose estimate is measured by the *Visible Surface
Discrepancy (VSD)* [1] with two settings of the misalignment tolerance in depth,
i.e. in the Z/optical axis: (1) *tau* = 20mm, (2) *tau* = infinity. With the
latter setting, VSD evaluates only the alignment of the visible surface mask and
is therefore more suitable for evaluation of RGB-only methods for which
estimating the Z component is more challenging.

The performance is measured by the *recall score*, i.e. the fraction of
annotated object instances for which a correct object pose was estimated. The
overall performance score is given by the average of the per-dataset recall
scores. We thus treat each dataset as a separate challenge and avoid the overall
score being dominated by larger datasets.

## How to Participate

### Challenge Rules

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

## Dates

...

## Organizers

...

## References

[1] Hodan, Michel et al.: BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.

[2] Hodan et al.: On Evaluation of 6D Object Pose Estimation, ECCVW'16.

[3] Newcombe et al.: KinectFusion: Real-time dense surface mapping and tracking,
    ISMAR'11.