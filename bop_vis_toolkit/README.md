# BOP Visualization Toolkit

A set of tools for visualizing your 6D pose estimation results on [BOP datasets](https://bop.felk.cvut.cz/datasets/).

## Available Visuals

- [x] rendering of object onto the image
- [x] depth error based on rendering with groundtruth and predicted poses
- [x] rendering of object symmetries
- [ ] depth error heatmaps based on rendering with groundtruth and predicted poses
- [ ] rendering of object contour
- [ ] 3D bounding boxes
- [ ] depth maps
- [ ] joint groundtruth-prediction visuals
- [ ] NOCS

## Installation

Please follow the steps in the [BOP Toolkit README](../README.md#installation) to install the BOP Toolkit first.

### Vispy Renderer (default)

The Python based headless renderer with egl backend is implemented using [Vispy](https://github.com/vispy/vispy).
Vispy is installed using the pip command above.
Note that the [nvidia opengl driver](https://developer.nvidia.com/opengl-driver) might be required in case of any errors.

### Python Renderer (deprecated)

Another Python based renderer is implemented using
[Glumpy](https://glumpy.github.io/) which depends on
[freetype](https://www.freetype.org/) and [GLFW](https://www.glfw.org/).
This implementation is similar to the vispy renderer since glumpy and vispy have similar apis,
but this renderer does not support headless rendering.
Glumpy is installed using the pip command above. On Linux, freetype and GLFW can
be installed by:

```
apt-get install freetype
apt-get install libglfw3
```

To install freetype and GLFW on Windows, follow [these instructions](https://glumpy.readthedocs.io/en/latest/installation.html#step-by-step-install-for-x64-bit-windows-7-8-and-10).

GLFW serves as a backend of Glumpy. [Another backend](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
can be used but were not tested with our code.

### C++ Renderer

For fast CPU-based rendering on a headless server, we recommend installing [bop_renderer](https://github.com/thodan/bop_renderer),
an off-screen C++ renderer with Python bindings.