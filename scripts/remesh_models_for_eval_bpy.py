"""
Batch remesh .ply model files using Blender's Voxel Remesh modifier.

Tested with Blender 4.5.10 LTS.

Command-line usage:
    blender --background --python scripts/remesh_models_for_eval_bpy.py -- <input_folder> <output_folder> [--voxel_size <voxel_size>]

Arguments:
    input_folder (mandatory): Path to the folder containing the input .ply files.
    output_folder (mandatory): Path to the folder where remeshed .ply files will be saved.
    voxel_size (optional): Voxel size (in mm) used for the remesh operation (default: 1).
"""

import bpy
import argparse
import os
import sys



def parse_args():
    """
    Parses command-line arguments passed after the '--' separator in a
    Blender command, e.g.:
        blender --background --python script.py -- <args>
    """
    # Blender passes its own arguments before '--'; everything after is ours.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Batch remesh .ply model files using Blender's Voxel Remesh modifier.",
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing the input .ply files.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where remeshed .ply files will be saved.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=1,
        help="Voxel size (in mm) used for the remesh operation (default: 1).",
    )
    return parser.parse_args(argv)


def clear_scene():
    """
    Clears all mesh objects from the current scene.
    """
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def process_files(input_folder, output_folder, voxel_size):
    """
    Main function to process all .ply files in the input folder.

    Args:
        input_folder (str): Path to the folder containing input .ply files.
        output_folder (str): Path to the folder where remeshed files will be saved.
        voxel_size (float): Voxel size in mm for the remesh operation.
    """

    print("--- Starting batch remesh process ---")
    print(f"Input folder:  '{input_folder}'")
    print(f"Output folder: '{output_folder}'")
    print(f"Voxel size:    {voxel_size} mm")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    files_in_dir = os.listdir(input_folder)
    files_to_process = [f for f in files_in_dir if f.lower().endswith(".ply")]
    if not files_to_process:
        print(f"Warning: No .ply files were found in '{input_folder}'. Please check the folder and file extensions.")
        return

    print(f"Found {len(files_to_process)} .ply file(s) to process.")

    for filename in files_to_process:
        clear_scene()
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"\nProcessing: {filename}")

        try:
            bpy.ops.wm.ply_import(filepath=input_path)
        except Exception as e:
            print(f"  - Failed to import {filename}. Error: {e}")
            continue

        obj = bpy.context.view_layer.objects.active
        if not obj or obj.type != 'MESH':
            # Handle cases where import succeeds but no object is active
            if len(bpy.context.scene.objects) > 0:
                for o in bpy.context.scene.objects:
                    if o.type == 'MESH':
                        obj = o
                        break
            if not obj:
                print(f"  - Could not find a valid mesh object after importing {filename}.")
                continue
            
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        print(f"  - Applying Voxel Remesh with size: {voxel_size}")
        remesh_mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
        remesh_mod.mode = 'VOXEL'
        remesh_mod.voxel_size = voxel_size
        remesh_mod.use_smooth_shade = True

        try:
            bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
        except Exception as e:
            print(f"  - Failed to apply modifier to {filename}. Error: {e}")
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        try:
            bpy.ops.wm.ply_export(filepath=output_path)
            print(f"  - Successfully saved to: {output_path}")
        except Exception as e:
            print(f"  - Failed to export {filename}. Error: {e}")

    clear_scene()
    print("\n--- Batch processing complete! ---")

if __name__ == "__main__":
    args = parse_args()
    process_files(args.input_folder, args.output_folder, args.voxel_size)