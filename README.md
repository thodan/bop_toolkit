# BOP Toolkit

A Python toolkit of the BOP benchmark for 6D object pose estimation
(http://bop.felk.cvut.cz).

- **bop_toolkit_lib** - The core Python library for i/o operations, calculation
  of pose errors, Python based rendering etc.
- **docs** - Documentation and conventions.
- **scripts** - Scripts for evaluation, rendering of training images,
  visualization of 6D object poses etc.

## Installation
Supported python versions: [3.8-3.12]

### Using [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```
This commands sets up a local venv (activate with `source .venv/bin/activate`), installs necessary dependencies and bop_toolkit_lib. You may provide additional flags such as:

- `--python 3.10`: specify the venv python version
- `--extra eval_coco`: install dependencies for coco evaluation
- `--extra eval_gpu`: install dependencies for gpu evaluation
- `--extra eval_hot3d`: install dependencies for hot3d evaluation
- `--extra scripts`: install dependencies for utility scripts (e.g. `annotation_tools.py`)

### Using pip
```bash
pip install .  # bop_toolkit_lib with core dependencies only
# with additional dependencies
pip install .[eval_coco]  # install dependencies for coco evaluation
pip install .[eval_gpu]  # install dependencies for gpu evaluation
pip install .[eval_hot3d]  # install dependencies for hot3d evaluation
uv pip install .[scripts]  # install dependencies for utility scripts (e.g. `annotation_tools.py`)
```

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

## Usage

### 1. Get the BOP datasets

Download the BOP datasets and make sure they are in the [expected folder structure](https://bop.felk.cvut.cz/datasets/).

### 2. Run your method

Estimate poses and save them in one .csv file per dataset ([format description](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#howtoparticipate)).

### 3. Configure the BOP Toolkit

In [bop_toolkit_lib/config.py](https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/config.py), set paths to the BOP datasets, to a folder with results to be evaluated, and to a folder for the evaluation output. The other parameters are necessary only if you want to visualize results or run the C++ Renderer.

### 4. Evaluate the pose estimates for 6D detection task
```
python scripts/eval_bop24_pose.py --result_filenames=NAME_OF_CSV_WITH_RESULTS --use_gpu
```
`--use_gpu`: Use GPU for the evaluation which requires [PyTorch]() installed and a GPU with CUDA support. The current implementation limits GPU memory usage to less than 2GB for BOP servers. If you have GPUs with larger memory, you can increase the limit by setting the [max_batch_size](https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error_gpu.py#L9) parameter. If GPU is not used, the evaluation is performed on CPU with 10 parallel processes. You can change the number of processes by setting the `--num_worker 1`.

`--result_filenames`: Comma-separated filenames with pose estimates in .csv ([examples](https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip)).

#### HOT3D special case
The [Hand Tracking Toolkit](https://github.com/facebookresearch/hand_tracking_toolkit) Fisheye camera implementation is necessary for evaluation on the HOT3D dataset. Install with:  

`pip install git+https://github.com/facebookresearch/hand_tracking_toolkit`

### 5. Evaluate the pose estimates for 6D localization task
```
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=NAME_OF_CSV_WITH_RESULTS
```
`--renderer_type`: "vispy", "python", or "cpp" (We recommend using "vispy" since it is easy to install and works headlessly. For "cpp", you need to install the C++ Renderer [bop_renderer](https://github.com/thodan/bop_renderer).).

`--result_filenames`: Comma-separated filenames with pose estimates in .csv ([examples](https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip)).

By default, this script is run with 10 parallel processes. You can change the number of processes by setting the `--num_worker 1`.

### 6. Evaluate the detections / instance segmentations
```
python scripts/eval_bop22_coco.py --result_filenames=NAME_OF_JSON_WITH_COCO_RESULTS --ann_type='bbox'
```
--result_filenames: Comma-separated filenames with per-dataset coco results (place them under your `results_path` defined in your [config.py](bop_toolkit_lib/config.py)).  
--ann_type: 'bbox' to evaluate amodal bounding boxes. 'segm' to evaluate segmentation masks.

## Convert BOP to COCO format

```
python scripts/calc_gt_coco.py
```

Set the dataset and split parameters in the top section of the script.

## Manual annotation tool

To annotate a new dataset in BOP format use [this tool](./scripts/annotation_tool.py).

Edit the file paths in parameters section at the beginning of the file then run:

```
python scripts/annotation_tool.py
```

### Interface:

Control the object pose with the following keys
`i`: up, `,`: down, `j`: front, `k`:back, `h`:left, `l`:right

Translation/rotation mode:
- Shift not clicked: translation mode
- Shift clicked: rotation model

Distance/angle big or small:
- Ctrl not clicked: small distance(1mm) / angle(2deg)
- Ctrl clicked: big distance(5cm) / angle(90deg)

R or "Refine" button will call ICP algorithm to do local refinement of the annotation
