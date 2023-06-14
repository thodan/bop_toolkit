# Training datasets for Tasks 4, 5 and 6 of the BOP Challenge 2023 [1]

The datasets include over 1M images showing more than 50K diverse objects. The images were originally synthesized for MegaPose [2] using BlenderProc [3]. The objects are from the Google Scanned Objects [4] and ShapeNetCore [5] datasets and their 3D models can be downloaded from the respective websites. Note that symmetry transformations are not available for these objects, but could be identified using these HALCON scripts [6] (we used the scripts to identify symmetries of objects in the BOP datasets [7] as described in Section 2.3 of the BOP Challenge 2020 paper [8]; if you use the scripts to identify symmetries of the Google Scanned Objects and ShapeNetCore objects, sharing the symmetries would be appreciated).

The datasets are saved in the BOP-webdataset format (see [8] for details).


## MegaPose-GSO dataset

Images of objects from Google Scanned Objects [4].

Mapping from `obj_id` to the original object identifiers:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/gso_models.json

Mapping from an image key to the index of the shard where it is stored:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/key_to_shard.json

The dataset is split into 1040 shards, with each shard containing 1000 images together with object annotations and camera parameters:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-000000.tar
...
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-001039.tar


## MegaPose-ShapeNetCore dataset

Images of objects from ShapeNetCore [5].

Mapping from `obj_id` to the original object identifiers:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/shapenet_models.json

Mapping from an image key to the index of the shard where it is stored:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/key_to_shard.json

The dataset is split into 1040 shards, with each shard containing 1000 images together with object annotations and camera parameters:
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-000000.tar
...
https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-001039.tar


## References

[1] https://bop.felk.cvut.cz/challenges/bop-challenge-2023/
[2] https://megapose6d.github.io/
[3] https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md
[4] https://research.google/resources/datasets/scanned-objects-google-research/
[5] https://shapenet.org/
[6] https://github.com/thodan/bop_toolkit/issues/50#issuecomment-903632625
[7] https://bop.felk.cvut.cz/datasets
[8] https://arxiv.org/pdf/2009.07378.pdf
[9] https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py
