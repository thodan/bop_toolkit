# SIXD Challenge 2017: format of results


For a test image XXXX and object YY, which is present in that image, the
evaluated method is expected to estimate the 6D pose of an instance of object YY
and save the results into file **XXXX\_YY.yml** with this format:

```
run_time: TIME
ests:
- {score: SCORE, R: [r11, r12, r13, r21, r22, r23, r31, r32, r33], t: [t1, t2, t3]}
```

where TIME is the run time (a single real number in seconds, -1 if not
available), SCORE is a confidence of the estimate (the range of SCORE is not
restricted), R = [r11 r12 r13; r21 r22 r23; r31 r32 r33] is a 3x3 rotation
matrix saved row-wise, and t = [t1 t2 t3]' is a 3x1 translation vector (in mm).
P = K * [R t] is the camera matrix that transforms 3D point p\_m = [x, y, z, 1]'
in the model coordinate system to 2D point p\_i = [u, v, 1]' in the image
coordinate system: s * p\_i = P * p\_m. The camera coordinate system is as
defined in OpenCV with the camera looking along the Z axis. K is provided with
the test images and might be different for each image.

We encourage participants to provide more estimates per file (i.e. more entries
in field "ests", with one entry per line). This will allow to calculate not only
the recall rate (i.e. the percentage of images for which a correct object pose
is estimated), but also e.g. the precision-recall curve or the top-N recall rate
(i.e. the percentage of images for which a correct object pose is among the N
estimates with the highest score).

All test images are used for the evaluation, even those with multiple instances
of the object of interest. Images with multiple instances are likely to be
easier and the number of instances can be seen as a parameter of the image that
indicates its difficulty, similarly as the level of occlusion or the amount of
clutter does. The list of objects that are present in an image can be obtained
from file gt.yml which is provided for each set of test images.

The files with results are expected in this structure:

**METHOD\_DATASET[\_TESTTYPE]/ZZ/XXXX\_YY.yml**

METHOD is the name of your method, DATASET is the dataset name {hinterstoisser,
tless, tudlight, rutgers, tejani, doumanoglou}, TESTTYPE is the type of test
images (currently used only for T-LESS where the possible types are {primesense,
kinect, canon} -- use primarily primesense for SIXD Challenge 2017), and ZZ,
XXXX, and YY is the ID of the test scene, the test image and the object. METHOD,
DATASET and TESTTYPE must not contain the underscore character "_".


### Example

Test image 0000 (let us consider the image from the Primesense sensor) of test
scene #1 from the T-LESS dataset contains objects #2, #25, #29 and #30. Your
method is expected to run four times on this image and save the results in
files:

- your-method_tless_primesense/01/0000\_02.yml
- your-method_tless_primesense/01/0000\_25.yml
- your-method_tless_primesense/01/0000\_29.yml
- your-method_tless_primesense/01/0000\_30.yml

Example content of file 0000_25.yml:
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

Sample results can be downloaded from:
http://ptak.felk.cvut.cz/6DB/public/sixd_results


### Documentation of results

If you want your results to be included in a publication about the challenge,
a documentation is required. It is expected in file
**METHOD\_DATASET[\_TESTTYPE]/eval_doc.txt** with this format:

```
SIXD Challenge 2017: documentation of results

Author: ...
Method: ... (related publication)
Parameter settings: ... (values of crucial method parameters)
Implementation: ... (link to an implementation, if available)
PC tech specs: ... (for comparison of run times)

Dataset: ...
Training input:
    3D model type: ... (none, default, for T-LESS: cad or reconst)
    Image type: ... (none, real, rendered)
    Image modalities: ... (none, RGB-D, RGB, D, grayscale)
    Viewpoint sampling: ... (if extra images were rendered)
    Image augmentation: ... (in-plane rotation, background, noise, etc.)
    Images per object: ... (counting also the augmented images)
    Object set: ... (specify the subset of objects whose training data was used;
                     typical is to use the training data of only the object of
                     interest or of all objects from the dataset)
Test input:
    Image modalities: ... (RGB-D, RGB, D)

Notes: ...
```
