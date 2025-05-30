# Training datasets for Tasks 4–6 of [BOP Challenge 2023](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/)

The datasets include over 2M images showing more than 50K diverse objects. The images were originally synthesized for [MegaPose](https://megapose6d.github.io/) using [BlenderProc](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md). The objects are from the [Google Scanned Objects](https://research.google/resources/datasets/scanned-objects-google-research/) and [ShapeNetCore](https://shapenet.org/) datasets and their 3D models can be downloaded from the respective websites.

Note that symmetry transformations are not available for these objects, but could be identified using [these HALCON scripts](https://github.com/thodan/bop_toolkit/issues/50#issuecomment-903632625) (we used the scripts to identify symmetries of objects in the [BOP datasets](https://bop.felk.cvut.cz/datasets) as described in Section 2.3 of the [BOP Challenge 2020 paper](https://arxiv.org/pdf/2009.07378.pdf); if you use the scripts to identify symmetries of the Google Scanned Objects and ShapeNetCore objects, sharing the symmetries would be appreciated).


## MegaPose-GSO dataset

- 3D object models can be downloaded from [Google Scanned Objects](https://research.google/resources/datasets/scanned-objects-google-research/). For the models to be compatible with GT poses, you will need to (i) center the objects and rescale them such that they fit within the unit sphere, and (ii) scale the normalized models by 0.1. Please see [this comment](https://github.com/thodan/bop_toolkit/issues/98#issuecomment-1718257952) for pseudo code.
- [Mapping from `obj_id` used in BOP to the original object identifiers](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-GSO/gso_models.json)
- [Mapping from an image key to the index of the shard where it is stored](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-GSO/key_to_shard.json)
- The dataset is in the [BOP-webdataset](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py) format and is split into 1040 shards, with each shard containing ~1000 images together with object annotations and camera parameters. Use the following URL template to download a shard (`<SHARD-ID>` is from `000000` to `001039`).
```
https://huggingface.co/datasets/bop-benchmark/megapose/tree/main/MegaPose-GSO/shard-<SHARD-ID>.tar
```


## MegaPose-ShapeNetCore dataset

- 3D object models can be downloaded from [ShapeNet](https://shapenet.org/) (scale the models by 0.1 to be compatible with GT poses)
- [Mapping from `obj_id` used in BOP to the original object identifiers](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-ShapeNetCore/shapenet_models.json)
- [Mapping from an image key to the index of the shard where it is stored](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-ShapeNetCore/key_to_shard.json)
- The dataset is in the [BOP-webdataset](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py) format and is split into 1040 shards, with each shard containing ~1000 images together with object annotations and camera parameters. Use the following URL template to download a shard (`<SHARD-ID>` is from `000000` to `001039`).
```
https://huggingface.co/datasets/bop-benchmark/megapose/tree/main/MegaPose-ShapeNetCore/shard-<SHARD-ID>.tar
```
