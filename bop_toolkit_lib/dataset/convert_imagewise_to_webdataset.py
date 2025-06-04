import argparse
import json
import multiprocessing
import pathlib
import re
import tarfile

import numpy as np
import webdataset as wds

from bop_toolkit_lib.dataset import bop_imagewise


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bop-imagewise -> bop-webdataset converter utility",
    )
    parser.add_argument(
        "--input",
        help="""A directory containing a dataset in imagewise format,
        e.g. ./ycbv/train_pbr_imwise.
        """,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="""Output directory that will contain
        multiple shards, e.g. ./ycbv/train_pbr_web
        """,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--maxcount",
        help="Maximum number of images per shard.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="""Randomly shuffle the images before creating
        the shards.""",
    )
    parser.add_argument(
        "--seed",
        help="Seed used for random shuffling.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--nprocs",
        help="""Number of subprocesses used.
        """,
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


def make_key_to_shard_map(
    wds_dir,
):
    key_to_shard = dict()
    for shard_path in wds_dir.glob("shard-*.tar"):
        shard_id = int(re.findall("\d+", shard_path.name)[0])
        names = tarfile.TarFile(shard_path).getnames()
        keys = set([name.split(".")[0] for name in names])
        for key in keys:
            key_to_shard[key] = shard_id
    return key_to_shard


def convert_imagewise_to_webdataset(
    input_dir,
    wds_dir,
    image_keys,
    start_shard,
    maxcount,
):
    wds_dir.mkdir(exist_ok=True)
    shard_writer = wds.ShardWriter(
        pattern=str(wds_dir / "shard-%06d.tar"),
        start_shard=start_shard,
        maxcount=maxcount,
        encoder=False,
    )
    infos = bop_imagewise.load_image_infos(input_dir, image_keys[0])

    for key in image_keys:

        def _file_path(ext):
            return input_dir / f"{key}.{ext}"

        obj = {
            "__key__": key,
        }

        if infos["has_rgb"]:
            rgb_name = "rgb" + infos["rgb_suffix"]
            obj[rgb_name] = open(_file_path(rgb_name), "rb").read()

        if infos["has_depth"]:
            obj["depth.png"] = open(_file_path("depth.png"), "rb").read()

        if infos["has_gray"]:
            obj["gray.tiff"] = open(_file_path("gray.tiff"), "rb").read()

        if infos["has_mask"]:
            obj["mask.json"] = open(_file_path("mask.json"), "rb").read()

        if infos["has_mask_visib"]:
            obj["mask_visib.json"] = open(_file_path("mask_visib.json"), "rb").read()

        if infos["has_gt"]:
            obj["gt.json"] = open(_file_path("gt.json"), "rb").read()

        if infos["has_gt_info"]:
            obj["gt_info.json"] = open(_file_path("gt_info.json"), "rb").read()

        obj["camera.json"] = open(_file_path("camera.json"), "rb").read()

        shard_writer.write(obj)


def main():
    args = parse_args()

    input_dir = pathlib.Path(args.input)
    wds_dir = pathlib.Path(args.output)

    input_file_paths = input_dir.glob("*")
    keys = set([p.name.split(".")[0] for p in input_file_paths])
    keys = list(keys)

    if args.shuffle:
        np.random.RandomState(args.seed).shuffle(keys)

    if args.nprocs > 0:
        keys_splits = np.array_split(keys, args.nprocs)
        _args = []
        start_shard = 0
        for keys_split in keys_splits:
            _args.append((input_dir, wds_dir, keys_split, start_shard, args.maxcount))
            n_shards = np.ceil(len(keys_split) / args.maxcount)
            start_shard += n_shards
        with multiprocessing.Pool(processes=args.nprocs) as pool:
            pool.starmap(convert_imagewise_to_webdataset, iterable=_args)
    else:
        convert_imagewise_to_webdataset(input_dir, wds_dir, keys, 0, args.maxcount)
    key_to_shard = make_key_to_shard_map(wds_dir)
    (wds_dir / "key_to_shard.json").write_text(json.dumps(key_to_shard))


if __name__ == "__main__":
    main()
