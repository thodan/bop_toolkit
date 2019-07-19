# BOP Toolkit

A Python toolkit of the BOP benchmark for 6D object pose estimation
(http://bop.felk.cvut.cz).

- **bop_toolkit_lib** - The core Python library for i/o operations, calculation
  of pose errors, Python based rendering etc.
- **docs** - Documentation and conventions.
- **scripts** - Scripts for evaluation, rendering of training images,
  visualization of 6D object poses etc.

## Installation

### Python Dependencies

To install the required python libraries, run:
```
pip install -r requirements.txt
```

In the case of problems, try to first run: ```pip install --upgrade pip setuptools```

### Python Renderer

The Python based renderer is implemented using
[Glumpy](https://glumpy.github.io/) which depends on
[freetype](https://www.freetype.org/) and [GLFW](https://www.glfw.org/).
Glumpy is installed using the pip command above. On Linux, freetype and GLFW can
be installed by:

```
apt-get install freetype
apt-get install libglfw3
```

To install freetype and GLFW on Windows, follow [these instructions](https://glumpy.readthedocs.io/en/latest/installation.html#step-by-step-install-for-x64-bit-windows-7-8-and-10).

GLFW serves as a backend of Glumpy. [Another backends](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
can be used but were not tested with our code.

### C++ Renderer

To speed up rendering, we recommend installing [bop_renderer](https://github.com/thodan/bop_renderer),
an off-screen C++ renderer with Python bindings.

See *scripts/eval_calc_errors.py* for an example on how to use the Python and
C++ renderers - you can switch between them by setting *renderer_type* to
*'python'* or *'cpp'*.
