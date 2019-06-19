# BOP Challenge 2019

## The Challenge Tasks

The methods are evaluated on the task of 6D localization of a single instance of
a single object. All test images are used for the evaluation, even those with
multiple instances of the object of interest.

Two tasks:

1. *The SiSo task* - 6D localization of a single instance of a single object.

2. *The ViVo task* - 6D localization of a varying number of instances of a varying number of objects.

## Submission instructions

1. Run your method on the SiSo task, i.e. the 6D localization of a single
instance of a single object [1], on all BOP datasets. Consider only a subset of
test images defined in files *test_set_v1.yml* provided with the datasets.
2. Prepare the results of your method in the format described below and submit
to the [BOP evaluation system](http://bop.felk.cvut.cz).

## Format of results

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


## Directory structure

The files with results are expected to have this structure:

*METHOD\_DATASET-SPLIT[\-TESTTYPE]/Z/Y\_X.yml*

where:
* METHOD is the name of the evaluated method.
* DATASET is the dataset name, one of {lm, lmo, tless, tudl, ruapc, icmi, icbin,
hb}.
* SPLIT shall be set to "test".
* TESTTYPE is the type of test images (currently used only for T-LESS where the
possible types are {primesense, kinect, canon}).
* Z is the scene ID.
* Y is the image ID.
* X is the object ID.

METHOD, DATASET, SPLIT and TESTTYPE must not contain the underscore character
"_".

## Subsets of test images

The ID's of test images used for the evaluation are in files *test_set_v1.yml*
provided with the datasets. The sets of test images from the original datasets
were reduced to remove redundancies and to encourage participation of new, in
particular slow, methods.

## Example

Test image 0 of test scene 1 from the T-LESS dataset contains objects 2, 25, 29
and 30. You are expected to provide four files with results of your method for
this image, each with pose estimates for one object (the zero padding of ID's is
not necessary):

- *your-method_tless-test-primesense/000001/000000\_000002.yml*
- *your-method_tless-test-primesense/000001/000000\_000025.yml*
- *your-method_tless-test-primesense/000001/000000\_000029.yml*
- *your-method_tless-test-primesense/000001/000000\_000030.yml*

Example content of file *000000_000025.yml*:
```
run_time: 1.6553
ests:
- {score: 2.591, R: [0.164305, -0.116608, -0.979493, -0.881643, 0.427981, -0.198842, 0.442391, 0.896233, -0.0324873], t: [-45.7994, 75.008, 801.078]}
- {score: 3.495, R: [-0.321302, 0.937843, -0.131205, -0.926472, -0.282636, 0.248529, 0.195999, 0.201411, 0.959697], t: [-77.2591, -23.8807, 770.016]}
- {score: 0.901, R: [0.133352, -0.546655, -0.826671, 0.244205, 0.826527, -0.507166, 0.960511, -0.134245, 0.243715], t: [79.4697, -23.619, 775.376]}
- {score: 1.339, R: [-0.998023, 0.0114256, 0.061813, -1.55661e-05, -0.983388, 0.181512, 0.0628601, 0.181152, 0.981445], t: [8.9896, 75.8737, 751.272]}
- {score: 1.512, R: [0.211676, 0.12117, -0.969799, 0.120886, 0.981419, 0.149007, 0.969835, -0.148776, 0.193096], t: [7.10206, -53.5385, 784.077]}
- {score: 0.864, R: [0.0414156, -0.024525, -0.998841, 0.295721, 0.955208, -0.0111921, 0.954376, -0.294915, 0.046813], t: [40.1253, -34.8206, 775.819]}
- {score: 1.811, R: [0.0369952, -0.0230957, -0.999049, 0.304581, 0.952426, -0.0107392, 0.951768, -0.303894, 0.0422696], t: [36.5109, -27.5895, 775.758]}
- {score: 1.882, R: [0.263059, -0.946784, 0.18547, -0.00346377, -0.193166, -0.98116, 0.964774, 0.25746, -0.0540936], t: [75.3467, -28.4081, 771.788]}
- {score: 1.195, R: [-0.171041, -0.0236642, -0.98498, -0.892308, 0.427616, 0.144675, 0.41777, 0.90365, -0.0942557], t: [-69.8224, 73.1472, 800.083]}
- {score: 1.874, R: [0.180726, -0.73069, 0.658354, 0.0538221, -0.661026, -0.74843, 0.98206, 0.170694, -0.0801374], t: [19.7014, -68.7299, 781.424]}
```

## References

[1] Hodan, Michel et al. "BOP: Benchmark for 6D Object Pose Estimation" ECCV'18.
