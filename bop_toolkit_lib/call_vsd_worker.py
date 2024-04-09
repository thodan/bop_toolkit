import argparse
import os
import time

from bop_toolkit_lib.renderer import create_renderer
from bop_toolkit_lib.pose_error import POSE_ERROR_VSD_ARGS
from bop_toolkit_lib import inout, misc, pose_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, help="worker id", required=True)
    parser.add_argument(
        "--input_dir", type=str, help="Path to the model file", required=True
    )
    args = parser.parse_args()

    worker_folder = os.path.join(args.input_dir, f"vsd_worker_{args.worker_id}")
    render_args_path = os.path.join(args.input_dir, "renderer_args.json")
    models_dict_path = os.path.join(args.input_dir, "models_dict.json")
    render_args = inout.load_json(render_args_path)
    models_dict = inout.load_json(models_dict_path)

    # init renderer
    renderer = create_renderer(
        render_args["width"],
        render_args["height"],
        render_args["render_type"],
        render_args["mode"],
        render_args["shading"],
        render_args["bg_color"],
    )
    # add objects
    for obj_id, model_path in models_dict.items():
        renderer.add_object(int(obj_id), model_path)

    start_time = time.time()
    all_files = os.listdir(worker_folder)

    results = {}
    for file in all_files:
        vsd_args = POSE_ERROR_VSD_ARGS.from_file(os.path.join(worker_folder, file))
        err = pose_error.vsd(
            vsd_args.R_e,
            vsd_args.t_e,
            vsd_args.R_g,
            vsd_args.t_g,
            vsd_args.depth_im,
            vsd_args.K,
            vsd_args.vsd_deltas,
            vsd_args.vsd_taus,
            vsd_args.vsd_normalized_by_diameter,
            vsd_args.diameter,
            renderer,
            vsd_args.obj_id,
            vsd_args.step,
        )
        results[file.split(".")[0]] = err
    save_path = os.path.join(args.input_dir, f"worker_{args.worker_id}.json")
    inout.save_json(save_path, results)
    misc.log(f"Worker {args.worker_id} finished in {time.time() - start_time} s")
