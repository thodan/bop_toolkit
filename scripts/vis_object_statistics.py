"""Visualizes statistics of models."""

import os
import argparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="xyzibd", help="Name of the BOP dataset. See dataset_params.py for options.")
parser.add_argument("--model_type", default=None, help="Type of object models to use. See dataset_params.py for options (e.g. cad, eval...). Default: None->'models' directory.")
parser.add_argument("--datasets_path", default=config.datasets_path, help="Path to the folder containing the BOP datasets.")
parser.add_argument("--vis_dir", default=os.path.join(config.output_path, "vis_object_statistics"), help="Path to the folder for output visualisations.")
parser.add_argument("--show", action="store_true", default=False, help="Show matplotlib plot.")
args = parser.parse_args()

vis_dir = Path(args.vis_dir)
vis_dir.mkdir(parents=True, exist_ok=True)

# Load dataset models.
dp_model = dataset_params.get_model_params(args.datasets_path, args.dataset, args.model_type)
obj_models = {}
for obj_id in tqdm(dp_model["obj_ids"], desc="Loading object models"):
    model_path = dp_model["model_tpath"].format(obj_id=obj_id)
    obj_models[obj_id] = trimesh.load(model_path, force="mesh")

# Collect triangle sizes for each object
obj_ids = dp_model["obj_ids"]
triangle_sizes = [np.sqrt(obj_models[obj_id].area_faces) for obj_id in obj_ids]

# Create box plot of triangle sizes
plt.figure(figsize=(10, 6))
plt.boxplot(triangle_sizes, tick_labels=obj_ids)
plt.title(f"dataset: {args.dataset}, model_type: {args.model_type}"  "\ntriangle sizes ($\sqrt{area}$) by Object ID")
plt.xlabel('Object ID')
plt.ylabel('Triangle size [mm]')
plt.grid(True)

# Save the plot
if args.show:
    plt.show()
else:
    save_path = vis_dir / f"{args.dataset}_model_{args.model_type}_triangle_sizes_boxplot.png"
    misc.log(f'Saving plot to: {save_path}')
    plt.savefig(save_path)
plt.close()
