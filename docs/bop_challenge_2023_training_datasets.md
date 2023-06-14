# Training datasets for Tasks 4--6 of the [BOP Challenge 2023](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/)

The datasets include over 1M images showing more than 50K diverse objects. The images were originally synthesized for [MegaPose](https://megapose6d.github.io/) using [BlenderProc](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md). The objects are from the [Google Scanned Objects](https://research.google/resources/datasets/scanned-objects-google-research/) and [ShapeNetCore](https://shapenet.org/) datasets and their 3D models can be downloaded from the respective websites. Note that symmetry transformations are not available for these objects, but could be identified using [these HALCON scripts](https://github.com/thodan/bop_toolkit/issues/50#issuecomment-903632625) (we used the scripts to identify symmetries of objects in the [BOP datasets](https://bop.felk.cvut.cz/datasets) as described in Section 2.3 of the [BOP Challenge 2020 paper](https://arxiv.org/pdf/2009.07378.pdf); if you use the scripts to identify symmetries of the Google Scanned Objects and ShapeNetCore objects, sharing the symmetries would be appreciated).

The datasets are saved in the [BOP-webdataset format](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py).


## MegaPose-GSO dataset

Images of objects from [Google Scanned Objects](https://research.google/resources/datasets/scanned-objects-google-research/).

Mapping from `obj_id` to the original object identifiers:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/gso_models.json

Mapping from an image key to the index of the shard where it is stored:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/key_to_shard.json

The dataset is split into 1040 shards, with each shard containing 1000 images together with object annotations and camera parameters:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-000000.tar
...
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-001039.tar


## MegaPose-ShapeNetCore dataset

Images of objects from [ShapeNetCore](https://shapenet.org/).

Mapping from `obj_id` to the original object identifiers:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/shapenet_models.json

Mapping from an image key to the index of the shard where it is stored:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/key_to_shard.json

The dataset is split into 1040 shards, with each shard containing 1000 images together with object annotations and camera parameters:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-000000.tar
...
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-001039.tar
