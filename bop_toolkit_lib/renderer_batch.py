# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)

"""Simple batch renderer for BOP toolkit, designed for parallel vsd computation
"""
import os
import numpy as np
import multiprocessing
import shutil

# Standard Library
from dataclasses import dataclass
from typing import Optional

from bop_toolkit_lib import inout, misc


@dataclass
class POSE_ERROR_VSD_ARGS:
    # all args in pose_error.vsd
    R_e: Optional[np.ndarray] = None
    t_e: Optional[np.ndarray] = None
    R_g: Optional[np.ndarray] = None
    t_g: Optional[np.ndarray] = None
    depth_im: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    vsd_deltas: Optional[float] = None
    vsd_taus: Optional[list] = None
    vsd_normalized_by_diameter: Optional[bool] = None
    diameter: Optional[float] = None
    obj_id: Optional[int] = None
    step: Optional[str] = None

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def to_file(self, path):
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError("Field {} is None".format(key))
            value = np.array(value)
        np.savez(path, self.__dict__)

    def from_file(path):
        args = POSE_ERROR_VSD_ARGS()
        data = np.load(path, allow_pickle=True)
        data = data["arr_0"].item()
        for key, value in data.items():
            if key == "vsd_taus":
                setattr(args, key, list(value))
            if key == "vsd_normalized_by_diameter":
                setattr(args, key, bool(value))
            if key == "step":
                setattr(args, key, str(value))
            if key == "obj_id":
                setattr(args, key, int(value))
            else:
                setattr(args, key, value)
        return args


class BatchRenderer:
    """
    Batch renderer for renderer.create_renderer
    """

    def __init__(
        self,
        width,
        height,
        renderer_type="cpp",
        mode="rgb+depth",
        shading="phong",
        bg_color=(0.0, 0.0, 0.0, 0.0),
        num_workers=1,
        tmp_dir="/tmp/vsd/",
    ):
        assert num_workers >= 1
        self.num_workers = num_workers
        self.tmp_dir = tmp_dir
        self.rendere_args = dict(
            width=width,
            height=height,
            render_type=renderer_type,
            mode=mode,
            shading=shading,
            bg_color=bg_color,
        )
        self.models = {}

    def add_object(self, obj_id, model_path):
        self.models[obj_id] = model_path

    def get_num_workers_used(self, all_im_errs):
        """
        for each worker, its is required to initiate the rendering buffer, load object model, from scratch
        which is slow. Thus, this multi-processing is only interesting when the number of images is large.
        We make a simple test to see if the number of images is large enough to use multi-processing:
        - If yes, it returns the number of workers available
        - If no, it returns 1
        """
        if len(all_im_errs) < self.num_workers * len(self.models.keys()):
            return 1
        else:
            return self.num_workers

    def _init_renderers(self, num_workers):
        # create folder for each worker
        for worker_id in range(num_workers):
            worker_folder = os.path.join(self.tmp_dir, f"vsd_worker_{worker_id}")
            os.makedirs(worker_folder, exist_ok=True)

        # save renderer args
        renderer_args_path = os.path.join(self.tmp_dir, "renderer_args.json")
        inout.save_json(renderer_args_path, self.rendere_args)

        # save model paths
        models_dict_path = os.path.join(self.tmp_dir, "models_dict.json")
        inout.save_json(models_dict_path, self.models)

    def run_vsd(self, all_im_errs):
        """
        Compute VSD errors for all POSE_ERROR_VSD_ARGS
        """
        # get number of workers used: 1 or self.num_workers
        num_workers_used = self.get_num_workers_used(all_im_errs)

        # call init_renderers for all workers, after adding all objects
        self._init_renderers(num_workers=num_workers_used)

        in_counter = 0
        for idx_im, im_errs in enumerate(all_im_errs):
            for idx_err, err in enumerate(im_errs):
                for gt_id in err["errors"].keys():
                    if isinstance(err["errors"][gt_id], POSE_ERROR_VSD_ARGS):
                        worker_id = in_counter % num_workers_used
                        vsd_args_path = os.path.join(
                            self.tmp_dir,
                            f"vsd_worker_{worker_id}/{idx_im}_{idx_err}_{gt_id}.npz",
                        )
                        err["errors"][gt_id].to_file(vsd_args_path)
                        in_counter += 1

        all_im_errs = self.start_run_vsd(all_im_errs, in_counter)
        return all_im_errs

    def start_run_vsd(self, all_im_errs, in_counter):
        # get number of workers used: 1 or self.num_workers
        num_workers_used = self.get_num_workers_used(all_im_errs)

        cmds = []
        for worker_id in range(num_workers_used):
            cmd = [
                "python",
                "bop_toolkit_lib/call_renderer.py",
                f"--input_dir={self.tmp_dir}",
                f"--worker_id={worker_id}",
            ]
            cmds.append(cmd)

        log_file_path = os.path.join(self.tmp_dir, "log.txt")
        log_file = misc.start_disable_output(log_file_path)
        with multiprocessing.Pool(num_workers_used) as pool:
            pool.map_async(misc.run_command, cmds)
            pool.close()
            pool.join()
        misc.stop_disable_output(log_file)

        out_counter = 0
        for worker_id in range(num_workers_used):
            worker_results = inout.load_json(self.tmp_dir + f"/worker_{worker_id}.json")
            for key, value in worker_results.items():
                idx_im, idx_err, gt_id = key.split("_")
                all_im_errs[int(idx_im)][int(idx_err)]["errors"][int(gt_id)] = value
                out_counter += 1
        assert out_counter == in_counter, "Number of input and output files mismatch"

        shutil.rmtree(self.tmp_dir)
        return all_im_errs
