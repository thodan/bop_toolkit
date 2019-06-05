# SIXD Challenge 2017: format of datasets


## Directory structure

The datasets have the following structure:

* **dataset_info.md** - Dataset-specific information.
* **camera.yml** - Camera parameters.
* **models[\_MODELTYPE]** - 3D object models.
* **train[\_TRAINTYPE]/YY/{rgb,depth,obj,seg}** - Training images of object YY.
* **test[\_TESTTYPE]/ZZ/{rgb,depth,mask}** - Test images of scene ZZ.
* **vis_gt_poses** - Visualizations of the ground truth object poses.

**MODELTYPE**, **TRAINTYPE** and **TESTTYPE** are optional and are used if more
data types are available.

The images are organized into subfolders:

* **rgb** - Color images.
* **depth** - Depth images (saved as 16-bit unsigned short, see camera.yml for
    the depth units).
* **obj** (optional) - Object coordinate images [4].
* **seg** (optional) - Segmentation masks of the objects (for training images).
* **mask** (optional) - Masks of the regions of interest (for test images).

The corresponding images across the subolders have the same ID,
e.g. **rgb/0000.png** and **depth/0000.png** is the color and the depth image
of the same RGB-D frame.


## Training and test images

Each set of training and test images is accompanied with file info.yml that
contains for each image the following information:

* **cam\_K** - 3x3 intrinsic camera matrix K (saved row-wise).
* **cam\_R\_w2c** (optional) - 3x3 rotation matrix R\_w2c (saved row-wise).
* **cam\_t\_w2c** (optional) - 3x1 translation vector t\_w2c.
* **view\_level** (optional) - Viewpoint subdivision level, see below.

The matrix K may be different for each image. For example, in the case of the
T-LESS dataset, the principal point is not constant because the provided images
were obtained by cropping a region around the origin of the world coordinate
system (i.e. the center of the turntable) in the captured images.

P\_w2i = K * [R\_w2c, t\_w2c] is the camera matrix which transforms 3D point
p\_w = [x, y, z, 1]' in the world coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_w2i * p\_w.

The ground truth object poses are provided in files gt.yml that contain for each
object in each image the following information:

* **obj\_id** - Object ID.
* **cam\_R\_m2c** - 3x3 rotation matrix R\_m2c (saved row-wise).
* **cam\_t\_m2c** - 3x1 translation vector t\_m2c.
* **obj\_bb** - 2D bounding box of projection of the 3D object model at the
    ground truth pose. It is given by (x, y, width, height), where (x, y) is the
    top-left corner of the bounding box. 

P\_m2i = K * [R\_m2c, t\_m2c] is the camera matrix which transforms 3D point
p\_m = [x, y, z, 1]' in the model coordinate system to 2D point p\_i =
[u, v, 1]' in the image coordinate system: s * p\_i = P\_m2i * p\_m.


## Acquisition of training images

The training images were obtained either by capturing of the real objects from
various viewpoints or by rendering of 3D object models.

The viewpoints, from which the objects were rendered, were sampled from a view
sphere as in [1] - by recursively subdividing an icosahedron. The level of
subdivision at which a viewpoint was added is saved in info.yml as view_level
(viewpoints corresponding to vertices of the icosahedron have view_level = 0,
viewpoints obtained in the first subdivision step have view_level = 1, etc.).
To reduce the number of viewpoints while preserving their uniform distribution
over the sphere surface, one can consider only viewpoints with view_level <= n,
where n is the highest considered level of subdivision.

For the rendering, the radius of the view sphere was set to the distance of the
closest occurrence of any modeled object over all test images. The distance was
calculated from the camera center to the origin of the model coordinate system.

The provided training images were rendered using this script:
https://github.com/thodan/sixd_toolkit/blob/master/tools/render_train_imgs.py


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

The camera coordinate system is as in OpenCV with the camera looking along the
Z axis:
http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html


## Camera parameters

Intrinsic camera parameters can be found in file camera.yml. However, these
parameters are meant only for simulation of the used sensor when rendering the
training images. Intrinsic camera parameters for individual images are in files
info.yml.

To the best of our knowledge, the image distortion parameters were estimated
only for the T-LESS dataset - the images were already processed to remove the
distortion, requiring no further action from the dataset user.


## Units

* Depth images: See camera.yml.
* 3D object models: 1 mm
* Translation vectors: 1 mm


## References

[1] Hinterstoisser et al. "Model based training, detection and pose estimation
    of texture-less 3d objects in heavily cluttered scenes" ACCV 2012.

[2] MeshLab, http://meshlab.sourceforge.net/.

[3] Thurrner and Wuthrich "Computing vertex normals from polygonal
    facets" Journal of Graphics Tools 3.1 (1998).

[4] Brachmann et al. "Learning 6d object pose estimation using 3d object
    coordinates." ECCV 2014.
