# BOP Toolkit

Python scripts to facilitate participation in the SIXD Challenge:
http://cmp.felk.cvut.cz/sixd/challenge_2017

- **conversion** - Scripts used to convert the datasets from the original format
                   to the SIXD standard format.
- **doc** - Documentation and conventions.
- **params** - Parameters (paths to datasets etc.) used by other scripts.
- **pysixd** - Core library that takes care of i/o operations, rendering,
               calculation of pose errors etc.
- **tools** - Scripts for evaluation, rendering of training images,
              visualization of 6D object poses etc.

## Installation

### Python Dependencies

To install the required python packages, run:

```
pip install -r requirements.txt
```

In the case of problems, try to run ```pip install --upgrade pip setuptools```
first.

### Renderers

The Python based renderer is implemented using the Glumpy library which depends
on the freetype and GLFW libraries. On Linux, the libraries can be installed by:

```
apt-get install freetype
apt-get install libglfw3
```

Instructions for Windows are [here](https://glumpy.readthedocs.io/en/latest/installation.html#step-by-step-install-for-x64-bit-windows-7-8-and-10).


GLFW serves as a backend of Glumpy. [Another backends](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
could be used but were not tested with our code.

## Evaluation

1. Run your method on the SIXD datasets and prepare the results in
[this format](https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_results_format.md).
2. Run **tools/eval_calc_errors.py** to calculate errors of the pose estimates
(fill list **result_paths** in the script with paths to the results first).
3. Run **tools/eval_loc.py** to calculate performance scores in the
6D localization task (fill list **error_paths** in the script with paths to the
calculated errors first).

- [Measuring error of 6D object pose](https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_measuring_error.pdf)
- [Sample results](http://ptak.felk.cvut.cz/6DB/public/sixd_results)
