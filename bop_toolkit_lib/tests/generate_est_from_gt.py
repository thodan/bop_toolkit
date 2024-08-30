# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Generating estimation from GT for debugging/unit tests purposes."""

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout


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
dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

complete_split = split
if dp_split["split_type"] is not None:
    complete_split += "_" + dp_split["split_type"]

lines = ["scene_id,im_id,obj_id,score,R,t,time"]

for scene_id in dp_split["scene_ids"]:
    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt = inout.load_scene_gt(dp_split["scene_gt_tpath"].format(scene_id=scene_id))
    scene_gt_info = inout.load_json(
        dp_split["scene_gt_info_tpath"].format(scene_id=scene_id), keys_to_int=True
    )
    # Go through each view in scene_gt
    for im_id, im_gt in scene_gt.items():
        im_gt_info = scene_gt_info[im_id]
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

path = f"./bop_toolkit_lib/tests/data/gt-pbrreal-rgb-mmodel_{dataset_name}-{split}_{dataset_name}.csv.csv"
with open(path, "w") as f:
    f.write("\n".join(lines))
