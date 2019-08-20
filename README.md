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

GLFW serves as a backend of Glumpy. [Another backend](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
can be used but were not tested with our code.

### C++ Renderer

For fast CPU-based rendering on a headless server, we recommend installing [bop_renderer](https://github.com/thodan/bop_renderer),
an off-screen C++ renderer with Python bindings.

In that repo you also find an example [renderer_test.py](https://github.com/thodan/bop_renderer/blob/master/samples/renderer_test.py) using the Python and the C++ renderer.

## Usage

### 1. Get the BOP datasets

Download the BOP datasets and make sure they are in the described folder structure:  
[BOP Challenge Datasets](https://bop.felk.cvut.cz/datasets/)

### 2. Run your method

Predict poses and save them in a .csv file per dataset. For details see:  
[How to participate](https://bop.felk.cvut.cz/challenges/bop_challenge_2019/#howtoparticipate)

### 3. Configure global config.py

In [bop_toolkit_lib/config.py](https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/config.py) set the paths to the BOP datasets, result folder to be evaluated and evaluation output.  

The other parameters are necessary if you want to visualize results or run the C++ Renderer.  

### 4. Run the full evaluation, e.g.
```
python scripts/eval_bop19.py --renderer_type=python --result_filenames=hodan-iros_lm-test.csv
```
--renderer_type: python / cpp  
--result_filenames: Comma-separated filenames with your pose results in .csv

If all data is in place the whole evaluation should run through. Otherwise, for additional optional parameters see: 
```
python scripts/eval_bop19.py -h
```
