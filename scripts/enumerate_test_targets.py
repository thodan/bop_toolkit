# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Creating the target list for the evaluation.

See docs/bop_datasets_format.md for documentation of the test target list.

The test target list, named test_targets_bop19.json is saved in the main folder of the selected dataset.
"""

import os

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "tudl",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Minimal visible fraction for a GT pose to be considered.
    "min_visib_fract": 0.1,
    # Name of file with a list of image ID's to be used. The file is assumed to be
    # stored in the dataset folder. None = all images are used for the evaluation.
    # 'im_subset_filename': 'test_set_v1.json',
    "im_subset_filename": None,
    # Name of the output file.
    "test_targets_filename": "test_targets_bop19.json",
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
)

# Subset of considered images.
if p["im_subset_filename"] is not None:
    im_ids_sets = inout.load_json(
        os.path.join(dp_split["base_path"], p["im_subset_filename"])
    )
else:
    im_ids_sets = None

# List of considered scenes.
scene_ids_curr = dp_split["scene_ids"]


test_targets = []
for scene_id in scene_ids_curr:
    misc.log("Processing scene: {}".format(scene_id))

    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)


    # Load the ground-truth poses.
    scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))

    # Load meta info about the ground-truth poses.
    scene_gt_info = inout.load_scene_gt(
        dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
    )

    # List of considered images.
    if im_ids_sets is not None:
        im_ids_curr = im_ids_sets[scene_id]
    else:
        im_ids_curr = sorted(scene_gt.keys())

    for im_id in im_ids_curr:
        # Find ID's of objects for which at least one instance is visible enough.
        obj_ids_visib_count = {}
        for gt_id, gt in enumerate(scene_gt[im_id]):
            if scene_gt_info[im_id][gt_id]["visib_fract"] >= p["min_visib_fract"]:
                if gt["obj_id"] not in obj_ids_visib_count:
                    obj_ids_visib_count[gt["obj_id"]] = 1
                else:
                    obj_ids_visib_count[gt["obj_id"]] += 1

        for obj_id, inst_count in obj_ids_visib_count.items():
            test_targets.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": obj_id,
                    "inst_count": inst_count,
                }
            )

# Save the test targets,
test_targets_path = os.path.join(dp_split["base_path"], p["test_targets_filename"])

misc.log("Saving {}".format(test_targets_path))
inout.save_json(test_targets_path, test_targets)

misc.log("Done.")
