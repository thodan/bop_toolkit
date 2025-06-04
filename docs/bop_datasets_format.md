# Format of BOP datasets

This file describes the [BOP-scenewise](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_scenewise.py) dataset format. This format can be converted to the [BOP-imagewise](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_imagewise.py) format using script [convert_scenewise_to_imagewise.py](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/convert_scenewise_to_imagewise.py) and to the [BOP-webdataset](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py) format using script [convert_imagewise_to_webdataset.py](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/convert_imagewise_to_webdataset.py).

Datasets provided on the [BOP website](https://bop.felk.cvut.cz/datasets) are in the BOP-scenewise format with exception of the [MegaPose training datasets](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_challenge_2023_training_datasets.md) provided for BOP Challenge 2023, which are in the BOP-webdataset format.


## Directory structure

The datasets have the following structure:

```
DATASET_NAME
├─ camera[_CAMTYPE].json
├─ dataset_info.json
├─ test_targets_bop19.json
├─ test_targets_bop24.json
├─ [test_targets_multiview_bop25.json]
├─ models[_MODELTYPE][_eval]
│  ├─ models_info.json
│  ├─ obj_OBJ_ID.ply
├─ train|val|test[_SPLITTYPE]|onboarding_static|onboarding_dynamic
│  ├─ SCENE_ID|OBJ_ID
│  │  ├─ scene_camera[_CAMTYPE].json
│  │  ├─ scene_gt[_CAMTYPE]son
│  │  ├─ scene_gt_info[_CAMTYPE].json
│  │  ├─ scene_gt_coco[_CAMTYPE].json
│  │  ├─ depth[_CAMTYPE]
│  │  ├─ mask[_CAMTYPE]
│  │  ├─ mask_visib[_CAMTYPE]
│  │  ├─ rgb|gray[_CAMTYPE]
```
[_SPLITTYPE] and [_CAMTYPE] are defined to be sensor and/or modality names in multi-sensory datasets.

* *models[\_MODELTYPE]* - 3D object models.
* *models[\_MODELTYPE]\_eval* - "Uniformly" resampled and decimated 3D object
  models used for calculation of errors of object pose estimates.


* *train[\_TRAINTYPE]/X* (optional) - Training images of object X.
* *val[\_VALTYPE]/Y* (optional) - Validation images of scene Y.
* *test[\_TESTTYPE]/Y* - Test images of scene Y.
* *onboarding_static/obj_X_SIDE* - Only for model-free tasks, static onboarding images of object X at up/down side.
* *onboarding_dynamic/obj_X* - Only for model-free tasks, dynamic onboarding images of object X.


* *camera.json* - Camera parameters (for sensor simulation only; per-image
  camera parameters are in files *scene_camera.json* - see below).
* *dataset_info.md* - Dataset-specific information.
* *test_targets_bop19.json* - A list of test targets used for the localization evaluation since the BOP Challenge 2019.
* *test_targets_bop24.json* - A list of test targets used for the detection evaluation since the BOP Challenge 2024.
* *test_targets_multiview_bop25.json* - A list of test targets used for the multi-view detection evaluation since the BOP Challenge 2025.


*MODELTYPE*, *TRAINTYPE*, *VALTYPE* and *TESTTYPE* are optional and used if more
data types are available (e.g. images from different sensors).

The images in *train*, *val* and *test* folders are organized into subfolders:

* *rgb/gray* - Color/gray images.
* *depth* - Depth images (saved as 16-bit unsigned short).
* *mask* (optional) - Masks of object silhouettes.
* *mask_visib* (optional) - Masks of the visible parts of object silhouettes.

The corresponding images across the subolders have the same ID, e.g.
*rgb/000000.png* and *depth/000000.png* is the color and the depth image
of the same RGB-D frame. The naming convention for the masks is IMID_GTID.png,
where IMID is an image ID and GTID is the index of the ground-truth annotation
(stored in *scene_gt.json*).


## Training, validation and test images

If both validation and test images are available for a dataset, the ground-truth
annotations are public only for the validation images. Performance scores for
test images with private ground-truth annotations can be calculated in the
[BOP evaluation system](http://bop.felk.cvut.cz).

### Camera parameters

Each set of images is accompanied with file *scene\_camera.json* which contains
the following information for each image:

* *cam\_K* - 3x3 intrinsic camera matrix K (saved row-wise).
* *depth_scale* - Multiply the depth image with this factor to get depth in mm.
* *cam\_R\_w2c* (optional) - 3x3 rotation matrix R\_w2c (saved row-wise).
* *cam\_t\_w2c* (optional) - 3x1 translation vector t\_w2c.
* *view\_level* (optional) - Viewpoint subdivision level, see below.

The matrix K may be different for each image. For example, the principal point
is not constant for images in T-LESS as the images were obtained by cropping a
region around the projection of the origin of the world coordinate system.

Note that the intrinsic camera parameters can be found also in file
*camera.json* in the root folder of a dataset. These parameters are meant only
for simulation of the used sensor when rendering training images.

P\_w2i = K * [R\_w2c, t\_w2c] is the camera matrix which transforms 3D point
p\_w = [x, y, z, 1]' in the world coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_w2i * p\_w.

### Ground-truth annotations

The ground truth object poses are provided in files *scene_gt.json* which
contain the following information for each annotated object instance:

* *obj\_id* - Object ID.
* *cam\_R\_m2c* - 3x3 rotation matrix R\_m2c (saved row-wise).
* *cam\_t\_m2c* - 3x1 translation vector t\_m2c.

P\_m2i = K * [R\_m2c, t\_m2c] is the camera matrix which transforms 3D point
p\_m = [x, y, z, 1]' in the model coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_m2i * p\_m.

Ground truth bounding boxes and instance masks are also provided in COCO format under *scene_gt_coco.json*. The RLE format is used for segmentations. Detailed information about the COCO format can be found [here](https://cocodataset.org/#format-data). 

### Meta information about the ground-truth poses

The following meta information about the ground-truth poses is provided in files
*scene_gt_info.json* (calculated using *scripts/calc_gt_info.py*, with delta =
5mm for ITODD, 15mm for other datasets, and 5mm for all photorealistic training
images provided for the BOP Challenge 2020):

* *bbox\_obj* - 2D bounding box of the object silhouette given by (x, y, width,
  height), where (x, y) is the top-left corner of the bounding box.
* *bbox\_visib* - 2D bounding box of the visible part of the object silhouette.
* *px\_count\_all* - Number of pixels in the object silhouette.
* *px\_count\_valid* - Number of pixels in the object silhouette with a valid
  depth measurement (i.e. with a non-zero value in the depth image).
* *px\_count\_visib* - Number of pixels in the visible part of the object
  silhouette.
* *visib\_fract* - The visible fraction of the object silhouette (= *px\_count\_visib*/*px\_count
_all*).


## Acquisition of training images

Most of the datasets include training images which were obtained either by
capturing real objects from various viewpoints or by rendering 3D object models
(using *scripts/render_train_imgs.py*).

The viewpoints, from which the objects were rendered, were sampled from a view
sphere as in [2] by recursively subdividing an icosahedron. The level of
subdivision at which a viewpoint was added is saved in *scene_camera.json* as
*view_level* (viewpoints corresponding to vertices of the icosahedron have
*view_level* = 0, viewpoints obtained in the first subdivision step have
*view_level* = 1, etc.). To reduce the number of viewpoints while preserving
their "uniform" distribution over the sphere surface, one can consider only
viewpoints with *view_level* <= n, where n is the highest considered level of
subdivision.

For rendering, the radius of the view sphere was set to the distance of the
closest occurrence of any annotated object instance over all test images. The
distance was calculated from the camera center to the origin of the model
coordinate system.


## 3D object models

The 3D object models are provided in PLY (ascii) format. All models include
vertex normals. Most of the models include also vertex color or vertex texture
coordinates with the texture saved as a separate image.
The vertex normals were calculated using
[MeshLab](http://meshlab.sourceforge.net/) as the angle-weighted sum of face
normals incident to a vertex [4].

Each folder with object models contains file *models_info.json*, which includes
the 3D bounding box and the diameter for each object model. The diameter is
calculated as the largest distance between any pair of model vertices.


## Coordinate systems

All coordinate systems (model, camera, world) are right-handed.
In the model coordinate system, the Z axis points up (when the object is
standing "naturally up-right") and the origin coincides with the center of the
3D bounding box of the object model.
The camera coordinate system is as in
[OpenCV](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
with the camera looking along the Z axis.


## Units

* Depth images: See files *camera.json/scene_camera.json* in individual
  datasets.
* 3D object models: 1 mm
* Translation vectors: 1 mm


## References

[1] Hodan, Michel et al. "BOP: Benchmark for 6D Object Pose Estimation" ECCV'18.

[2] Hinterstoisser et al. "Model based training, detection and pose estimation
    of texture-less 3d objects in heavily cluttered scenes" ACCV'12.

[3] Thurrner and Wuthrich "Computing vertex normals from polygonal
    facets" Journal of Graphics Tools 3.1 (1998).
