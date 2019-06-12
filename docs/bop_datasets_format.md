# BOP: format of datasets

This file describes the common format of the BOP datasets.


## Directory structure

The datasets have the following structure:


* *models[\_MODELTYPE]* - 3D object models.


* *train[\_TRAINTYPE]/X* (optional) - Training images of object X.
* *val[\_VALTYPE]/Y* (optional) - Validation images of scene Y.
* *test[\_TESTTYPE]/Y* - Test images of scene Y.


* *train[\_TRAINTYPE]\_gt\_stats* (optional) - Stats of the GT poses in training
  images.
* *val[\_TRAINTYPE]\_gt\_stats* (optional) - Stats of the GT poses in validation
  images.
* *test[\_TRAINTYPE]\_gt\_stats* - Stats of the GT poses in test images.


* *camera.yml* - Camera parameters.
* *dataset_info.md* - Dataset-specific information.
* *test_set_[VERSION].yml* - A subset of test images for evaluation.


*MODELTYPE*, *TRAINTYPE* and *TESTTYPE* are optional and are used if more
data types are available (e.g. images from different sensors).

The images in *train*, *val* and *test* folders are organized into subfolders:

* *rgb* - Color images.
* *depth* - Depth images (saved as 16-bit unsigned short).
* *mask* (optional) - Masks of object silhouettes.
* *mask_visib* (optional) - Masks of the visible parts of object silhouettes.

The corresponding images across the subolders have the same ID, e.g.
*rgb/000000.png* and *depth/000000.png* is the color and the depth image
of the same RGB-D frame.


## Training, validation and test images

Each set of images is accompanied with file *info.yml* that contains for each
image the following information:

* *cam\_K* - 3x3 intrinsic camera matrix K (saved row-wise).
* *depth_scale* - Multiply the depth image with this factor to get depth in mm.
* *cam\_R\_w2c* (optional) - 3x3 rotation matrix R\_w2c (saved row-wise).
* *cam\_t\_w2c* (optional) - 3x1 translation vector t\_w2c.
* *view\_level* (optional) - Viewpoint subdivision level, see below.

The matrix K may be different for each image. For example, the principal point
is not constant for images in T-LESS as the images were obtained by cropping a
region around the projection of the origin of the world coordinate system.

P\_w2i = K * [R\_w2c, t\_w2c] is the camera matrix which transforms 3D point
p\_w = [x, y, z, 1]' in the world coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_w2i * p\_w.

The ground truth object poses are provided in files *gt.yml* that contain for
each object in each image the following information:

* *obj\_id* - Object ID.
* *cam\_R\_m2c* - 3x3 rotation matrix R\_m2c (saved row-wise).
* *cam\_t\_m2c* - 3x1 translation vector t\_m2c.

P\_m2i = K * [R\_m2c, t\_m2c] is the camera matrix which transforms 3D point
p\_m = [x, y, z, 1]' in the model coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_m2i * p\_m.

If both validation and test images are available for a dataset, the ground-truth
annotations are public only for the validation images. Performance scores for
test images with private ground-truth annotations can be calculated at
[bop.felk.cvut.cz](http://bop.felk.cvut.cz).


## Stats of the ground-truth poses

The folders *\*_gt_stats* contain one file per scene with statistics of the
ground-truth poses calculated using *scripts/calc_gt_stats.py*:

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
sphere as in [1] by recursively subdividing an icosahedron. The level of
subdivision at which a viewpoint was added is saved in *info.yml* as
*view_level* (viewpoints corresponding to vertices of the icosahedron have
*view_level* = 0, viewpoints obtained in the first subdivision step have
*view_level* = 1, etc.). To reduce the number of viewpoints while preserving
their "uniform" distribution over the sphere surface, one can consider only
viewpoints with *view_level* <= n, where n is the highest considered level of
subdivision.

For rendering, the radius of the view sphere was set to the distance of the
closest occurrence of any object over all test images. The distance was
calculated from the camera center to the origin of the model coordinate system.


## 3D object models

The 3D object models are provided in PLY (ascii) format. All the models include
vertex normals. Most of the models include also vertex color or vertex texture
coordinates with the texture saved as a separate image.

The vertex normals were calculated using MeshLab [2] as the angle-weighted sum
of face normals incident to a vertex [3].


## Coordinate systems

All coordinate systems (model, camera, world) are right-handed.

In the model coordinate system, the Z axis points up (when the object is
standing "naturally up-right") and the origin coincides with the center of the
3D bounding box of the object model.

The camera coordinate system is [as in OpenCV](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
with the camera looking along the Z axis.


## Camera parameters

Intrinsic camera parameters can be found in file *camera.yml*. However, these
parameters are meant only for simulation of the used sensor when rendering the
training images. Intrinsic camera parameters for individual images are in files
*info.yml*.


## Units

* Depth images: See *camera.yml* of individual datasets.
* 3D object models: 1 mm
* Translation vectors: 1 mm


## References

[1] Hinterstoisser et al. "Model based training, detection and pose estimation
    of texture-less 3d objects in heavily cluttered scenes" ACCV 2012.

[2] MeshLab, http://meshlab.sourceforge.net/.

[3] Thurrner and Wuthrich "Computing vertex normals from polygonal
    facets" Journal of Graphics Tools 3.1 (1998).
