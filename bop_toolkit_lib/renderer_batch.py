# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)

"""Simple batch renderer for BOP toolkit, designed for parallel vsd computation
"""
import os

import multiprocessing
import shutil
from bop_toolkit_lib import inout, misc
from bop_toolkit_lib.pose_error import POSE_ERROR_VSD_ARGS


class BatchRenderer:
    """
    Batch renderer for renderer.create_renderer
    """

    def __init__(
        self,
        width,
        height,
        renderer_type="vispy",
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

        cmds = []
        for worker_id in range(num_workers_used):
            cmd = [
                "python",                
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "call_vsd_worker.py"),
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
