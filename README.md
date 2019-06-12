# BOP Toolkit

A Python toolkit for the BOP benchmark on 6D object pose estimation
(http://bop.felk.cvut.cz).

- **bop_toolkit** - The core Python library for i/o operations, rendering,
  calculation of pose errors etc.
- **docs** - Documentation and conventions.
- **scripts** - Scripts for evaluation, rendering of training images,
  visualization of 6D object poses etc.

## Installation

### Python Dependencies

To install the required python libraries, run:
```
pip install -r requirements.txt
```

In the case of problems, try to run ```pip install --upgrade pip setuptools```
first.

### Python Renderer

The Python based renderer is implemented using the Glumpy library which depends
on the freetype and GLFW libraries. On Linux, the libraries can be installed by:

```
apt-get install freetype
apt-get install libglfw3
```

[Installation instructions for Windows](https://glumpy.readthedocs.io/en/latest/installation.html#step-by-step-install-for-x64-bit-windows-7-8-and-10).

GLFW serves as a backend of Glumpy. [Another backends](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
can be used but were not tested with our code.

### C++ Renderer

To speed up rendering, we recommend installing [bop_renderer](https://github.com/thodan/bop_renderer),
an off-screen C++ renderer with Python bindings.

See *scripts/eval_calc_errors.py* for an example on how to use the Python and
C++ renderers - you can swith between them by setting *renderer_type* to
*'python'* or *'cpp'*.

## Evaluation in BOP Challenge 2019

The evaluation in the [BOP evaluation system](http://bop.felk.cvut.cz) is done
in two steps:

1. Errors of pose estimates are calculated using
*scripts/eval_calc_errors.py*. The pose estimates are expected in
[this format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_challenge_2019_results_format.md)
([samples](http://ptak.felk.cvut.cz/6DB/public/bop_sample_results)).
2. Performance scores are calculated using *scripts/eval_calc_scores.py* (using
the errors calculated in the first step).

For details about the evaluation methodology, see:  
*Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation", ECCV 2018.*
