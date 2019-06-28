# BOP Challenge 2019

## The Challenge Task

The challenge is on the task of 6D localization of a *varying* number of
instances of a *varying* number of objects in a single RGB-D image (the ViVo
task).

**Training Input:** At training time, method *M* learns using a training set,
*T = {T<sub>o</sub>}*, where *o* is an object identifier. Training data
*T<sub>o</sub>* may have different forms – a 3D mesh model of the object or a
set of RGB-D images (synthetic or real) showing object instances in known 6D
poses.

**Test Input:** At test time, method *M* is provided image *I* and list
*L = [(o<sub>1</sub>, n<sub>1</sub>), ..., (o<sub>m</sub>, n<sub>m</sub>)]*,
where *n<sub>i</sub>* is the number of instances of object *o<sub>i</sub>*
present in image *I*.

**Test Output:** Method *M* produces list
*E = [E<sub>1</sub>, ..., E<sub>m</sub>]*, where *E<sub>i</sub>* is a list of
*n<sub>i</sub>* pose estimates for instances of object *o<sub>i</sub>*. Each
estimate is given by a 3x3 rotation matrix, *R*, a 3x1 rotation vector, *t*,
and a confidence score, *s*.

The ViVo task is referred to as the *6D localization problem* in [2]. In the BOP
paper [1], methods were evaluated on a different task – 6D localization of a
single instance of a single object (the SiSo task). The simpler SiSo task was
chosen because it allowed to evaluate all relevant methods out of the box.
However, the state of the art has advanced and we are moving to the
more challenging ViVo task.

## Datasets

**Content of the Datasets:** The datasets include 3D object models and training
and test RGB-D images annotated with ground-truth 6D object poses and intrinsic
camera parameters. The 3D object models were created manually or using
KinectFusion-like systems for 3D surface reconstruction [3].

**Training and Test Images:** The training images show individual objects from
different viewpoints and are either captured by an RGB-D/Gray-D sensor or
obtained by rendering of the 3D object models. The test images were captured in
scenes with graded complexity, often with clutter and occlusion.

**Rules for Training:** For training, method *M* can use the provided object
models and training images, but can also render extra training images using the
object models. Not a single pixel of test images may be used in training, nor
the individual ground-truth poses provided for the test images. The distribution
of the ground-truth poses in the test images (provided in
*bop_toolkit/dataset_params.py*) is the only information about the test set that
can be used in training.

**The Core Datasets:** Method *M* needs to be evaluated on these 7 datasets to
be considered for the main challenge awards (see below).

* **LM-O** (Linemod-Occluded) [4,5]
* **T-LESS** [6]
* **TUD-L** (TU Dresden Light) [1]
* **IC-BIN** (Doumanoglou et al.) [7]
* **ITODD** (MVTec ITODD) [8]
* **HB** (HomebrewedDB) [9]
* **YCB-V** (YCB-Video) [10]

**Other datasets:**

* **LM** (Linemod) [4]
* **RU-APC** (Rutgers APC) [11]
* **IC-MI** (Tejani et al. [12]
* **TYO-L** (Toyota Light) [1]
* **DLR**

The datasets (in the
[BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md))
are available in the
[BOP evaluation system](http://bop.felk.cvut.cz/datasets/).

## Awards

The following awards will be presented at the
[5th International Workshop on Recovering 6D Object Pose](http://cmp.felk.cvut.cz/sixd/workshop_2019/)
at ICCV 2019:

1. **The Best Method** (prize money: X EUR) – The top-ranked method on the 7
core datasets.
2. **The Best RGB-Only Method** (prize money: X EUR) – The top-ranked RGB-only
method on the 7 core datasets.
3. **The Best Open Source Method** (prize money: Y EUR) – The top-ranked method
on the 7 core datasets whose source code is publicly available.
4. **The Best Method on Dataset D** – The top-ranked method on each of the 12
available datasets.

## Evaluation Methodology

**Measuring Error:** The error of a 6D object pose estimate is measured by three
pose error functions:

1. **Visible Surface Discrepancy (*e<sub>VSD</sub>*)** [1] with misalignment
tolerance *tau* = 20mm and threshold of correctness *theta<sub>VSD</sub>* = 0.3.
2. **Complement over Union of Object Silhouettes (*e<sub>CUS</sub>*)** with
threshold of correctness *theta<sub>CUS</sub>* = 0.3. The silhouettes are
obtained by rendering the object model in the estimated and the ground-truth
poses.
3. **Average Distance of Model Vertices (*e<sub>AD</sub>*)** [2,4] with
threshold of correctness *theta<sub>AD</sub>* = *0.1 * d*, where *d* is the
object diameter, i.e. the largest distance between any pair of model vertices.
The distance is measured to the same vertex if the object has no
indistinguishable views (e<sub>ADD</sub>) and to the closest vertex otherwise
(e<sub>ADI</sub>).

Functions *e<sub>AD</sub>* and *e<sub>VSD</sub>* are commonly used in the
literature. *e<sub>CUS</sub>* is suitable for evaluation of RGB-only methods for
which estimating the object position along the Z axis, i.e. the optical axis, is
more challenging as the projected object size hardly changes within the scope of
a few centimeters.

**Performance Score:** The performance w.r.t. pose error function
*e<sub>X</sub>*, where *X* is *VSD*, *CUS* or *AD*, is measured by the
*recall score*, i.e. the fraction of annotated object instances for which a
correct object pose was estimated. A pose estimate is considered correct if the
value of *e<sub>X</sub>* is below threshold of correctness *theta<sub>X</sub>*.

**Ranking of Methods:** For each dataset, three recall scores (one for each pose
error function) are calculated for each method. The ranking of methods on a
dataset is calculated by ordering the sums of ranks of the three recall scores.
The overall ranking of methods on the 7 core datasets is then calculated by
ordering the sums of the per-dataset ranks. This follows the ranking approach
used in the
[Robust Vision Challenge](http://www.robustvision.net/leaderboard.php).

## How to Participate

To have your method evaluated, run it on the ViVo task and submit the results in
the format described below to the
[BOP evaluation system](http://bop.felk.cvut.cz).

**Format of Results:** Results for all test images from one dataset are saved in
one CSV file, with one pose estimate per line in the following format:

```
scene_id,im_id,obj_id,score,R,t,time
```

* *scene_id*, *im_id*, and *obj_id* is the ID of scene, image and object
respectively.
* *score* is a confidence of the estimate (the range of confidence values is not
restricted).
* *R* is a 3x3 rotation matrix whose elements are saved row-wise and separated
by a white space (i.e. ```r11 r12 r13 r21 r22 r23 r31 r32 r33```, where
*r<sub>ij</sub>* is an element from the *i*-th row and the *j*-th column of the
matrix).
* *t* is a 3x1 translation vector (in mm) whose elements are separated by a
white space (i.e. ```t1 t2 t3```).
* *time* is the time method *M* took to make estimates for all objects in image
*im_id* from scene *scene_id*. All estimates with the same *scene_id* and
*im_id* must have the same value of *time*. Report the wall time from the point
right after the raw data (the image, 3D object models etc.) is loaded to the
point when the final pose estimates are available (a single real number in
seconds, -1 if not available).

*P = K * [R t]* is the camera matrix that transforms 3D point
*p<sub>m</sub> = [x, y, z, 1]'* in the model coordinate system to 2D point
*p<sub>i</sub> = [u, v, 1]'* in the image coordinate system:
*s * p<sub>i</sub> = P * p<sub>m</sub>*. The camera coordinate system is defined
as in OpenCV with the camera looking along the *Z* axis. Camera matrix *K* is
provided with the test images and might be different for each image.

**List of Object Instances:** The list of object instances for which the pose is
to be estimated can be found in files *test_targets_bop19.yml* provided with the
datasets. The list includes instances which are visible from at least 10% [1].

**Publication:** If you want your results to be included in a publication about
the challenge, a documentation of the method, including the tech specs of the
used computer, needs to be provided through the online submission form. Without
the documentation, your results will be listed on the website but not included
in the publication.

Example results can be found
[here](http://ptak.felk.cvut.cz/6DB/public/bop_sample_results).

## Dates

* Deadline for submission of results: **October 14, 2019** (11:59PM PST)
* Presentation of awards: **October 28, 2019**
(at the [ICCV 2019 workshop](http://cmp.felk.cvut.cz/sixd/workshop_2019/))

## References

[1] Hodan, Michel et al.: BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.

[2] Hodan et al.: On Evaluation of 6D Object Pose Estimation, ECCVW'16.

[3] Newcombe et al.: KinectFusion: Real-time dense surface mapping and tracking,
    ISMAR'11.

[4] Hinterstoisser et al.: Model based training, detection and pose estimation
    of texture-less 3d objects in heavily cluttered scenes, ACCV'12.

[5] Brachmann et al.: Learning 6d object pose estimation using 3d object
    coordinates, ECCV'14.
    
[6] Hodan et al.: T-LESS: An RGB-D Dataset for 6D Pose Estimation of
    Texture-less Objects, WACV'17.

[7] Doumanoglou et al.: Recovering 6D Object Pose and Predicting Next-Best-View
    in the Crowd, CVPR'16.

[8] Drost et al.: Introducing MVTec ITODD - A Dataset for 3D Object Recognition
    in Industry, ICCVW'17.

[9] Kaskman et al.: HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D
    Objects, arXiv:1904.03167

[10] Xiang et al.: PoseCNN: A Convolutional Neural Network for 6D Object Pose
     Estimation in Cluttered Scenes, RSS'18.
     
[11] Rennie et al.: A dataset for improved rgbd-based object detection and pose
     estimation for warehouse pick-and-place, Robotics and Automation
     Letters 2016.
 
[12] Tejani et al.: Latent-class hough forests for 3D object detection and pose
     estimation, ECCV'14.
