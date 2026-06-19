"""Visualizes statistics of models."""

import os
import argparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc


def get_face_sizes(mesh):
    """
    Extracts the number of vertices per face for the original loaded PLY mesh.
    Returns a numpy array of face sizes, or None if no face data exists.
    """
    if "_ply_raw" not in mesh.metadata:
        if hasattr(mesh, "faces") and len(mesh.faces) > 0:
            return np.full(len(mesh.faces), 3, dtype=int)
        return np.array([], dtype=int)
    raw = mesh.metadata["_ply_raw"]
    if "face" not in raw or not raw["face"]["length"]:
        return np.array([], dtype=int)
    
    face_data = raw["face"]["data"]
    index_names = ["vertex_index", "vertex_indices"]
    
    if isinstance(face_data, dict):
        faces = None
        for name in index_names:
            if name in face_data:
                faces = face_data[name]
                break
        if faces is None:
            return None
        
        if len(faces.shape) == 2:
            return np.full(faces.shape[0], faces.shape[1], dtype=int)
        elif faces.dtype == object or len(faces.shape) == 1:
            return np.array([len(f) for f in faces], dtype=int)
            
    elif isinstance(face_data, np.ndarray):
        name = None
        if len(face_data.dtype.names) == 1:
            name = face_data.dtype.names[0]
        elif len(face_data.dtype.names) > 1:
            for n in index_names:
                if n in face_data.dtype.names:
                    name = n
                    break
        if name is None:
            return None
        
        sub_array = face_data[name]
        if sub_array.dtype.names and 'f0' in sub_array.dtype.names:
            return sub_array['f0'].astype(int)
        
    return None


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
for obj_id in dp_model["obj_ids"]:
    model_path = dp_model["model_tpath"].format(obj_id=obj_id)
    mesh = trimesh.load(model_path, force="mesh")
    obj_models[obj_id] = mesh

    # Count the proportion of types of faces (triangles, quads)
    face_sizes = get_face_sizes(mesh)
    if face_sizes is not None and len(face_sizes) > 0:
        total_faces = len(face_sizes)
        num_triangles = np.sum(face_sizes == 3)
        num_quads = np.sum(face_sizes == 4)
        num_others = total_faces - num_triangles - num_quads

        prop_triangles = num_triangles / total_faces
        prop_quads = num_quads / total_faces
        prop_others = num_others / total_faces

        misc.log(
            f"Object {obj_id}: total faces = {total_faces}, "
            f"triangles = {num_triangles} ({prop_triangles:.2%}), "
            f"quads = {num_quads} ({prop_quads:.2%}), "
            f"others = {num_others} ({prop_others:.2%})"
        )

        if num_triangles < total_faces:
            misc.log(
                f"Warning: Object {obj_id} has non-triangular faces! "
                f"({total_faces - num_triangles} non-triangle faces)"
            )

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
