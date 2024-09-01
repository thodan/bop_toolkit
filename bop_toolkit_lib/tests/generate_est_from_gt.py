# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Generating estimation from GT for debugging/unit tests purposes."""

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
import os

# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "lmo",
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "test",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Minimum visibility of the GT poses to include them in the output.
    "min_visib_gt": 0.1,
    # File with the GT poses to consider.
    "targets_filename": "test_targets_bop19.json",
}
################################################################################

datasets_path = p["datasets_path"]
dataset_name = p["dataset"]
split = p["dataset_split"]
split_type = p["dataset_split_type"]
min_visib_gt = p["min_visib_gt"]

dp_split = dataset_params.get_split_params(
    datasets_path, dataset_name, split, split_type=split_type
)
# Load the targets to consider.
targets = inout.load_json(
    os.path.join(dp_split["base_path"], p["targets_filename"])
)
targets_org = {}
for target in targets:
    targets_org.setdefault(target["scene_id"], {})[target["im_id"]] = target

lines = ["scene_id,im_id,obj_id,score,R,t,time"]

total_number_instances = 0
for scene_id, scene_targets in targets_org.items():
    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt = inout.load_scene_gt(dp_split["scene_gt_tpath"].format(scene_id=scene_id))
    scene_gt_info = inout.load_json(
        dp_split["scene_gt_info_tpath"].format(scene_id=scene_id), keys_to_int=True
    )
    for im_id, im_targets in scene_targets.items():
        im_gt_info = scene_gt_info[im_id]
        im_gt = scene_gt[im_id]
        for idx_obj in range(len(im_gt)):
            obj_gt = im_gt[idx_obj]
            obj_gt_info = im_gt_info[idx_obj]

            if obj_gt_info["visib_fract"] < min_visib_gt:
                print(
                    f"Skipping object {obj_gt['obj_id']} in scene {scene_id}, image {im_id}"
                )
                continue
            # Load the GT pose.
            R = obj_gt["cam_R_m2c"]
            t = obj_gt["cam_t_m2c"]
            score = 1
            time = 1

            lines.append(
                "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_gt["obj_id"],
                    score=score,
                    R=" ".join(map(str, R.flatten().tolist())),
                    t=" ".join(map(str, t.flatten().tolist())),
                    time=time,
                )
            )
            total_number_instances += 1

path = f"./bop_toolkit_lib/tests/data/gt-pbrreal-rgb-mmodel_{dataset_name}-{split}_{dataset_name}.csv"
with open(path, "w") as f:
    f.write("\n".join(lines))
print(f"Generated {total_number_instances} instances to {path}")