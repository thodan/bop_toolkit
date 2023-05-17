import argparse
import itertools
import multiprocessing
import pathlib
import shutil

import tqdm

from bop_toolkit_lib.dataset import bop_v1, bop_v2


def convert_v1_scene_to_v2(
    v1_scene_dir,
    v2_dir
):
    """Converts a scene in v1 format to v2 format.

    :param v1_scene_dir:
    Directory containing observations and annotations in v1 format.
    :param v2_dir: Directory where the data will be written in v2 format.
    """
    scene_id = int(v1_scene_dir.name)

    scene_data = bop_v1.load_scene_data(
        v1_scene_dir,
        load_scene_camera=True,
        load_scene_gt=True,
        load_scene_gt_info=True,
    )

    scene_infos = bop_v1.read_scene_infos(
        v1_scene_dir,
    )

    v2_dir.mkdir(exist_ok=True)

    image_tkey = f"{scene_id:06d}_" + "{image_id:06d}"

    bop_v2.save_scene_camera(
        scene_data["scene_camera"],
        v2_dir / (image_tkey + ".camera.json")
    )

    bop_v2.save_scene_gt(
        scene_data["scene_gt"], v2_dir / (image_tkey + ".gt.json")
    )

    bop_v2.save_scene_gt(
        scene_data["scene_gt_info"],
        v2_dir / (image_tkey + ".gt_info.json")
    )

    image_ids = [int(k) for k in scene_data["scene_camera"].keys()]
    for image_id in image_ids:
        image_key = image_tkey.format(image_id=image_id)
        for mask_type in ("mask", "mask_visib"):
            if scene_infos["has_" + mask_type]:
                masks = bop_v1.load_masks(
                    v1_scene_dir,
                    image_id, mask_type=mask_type)
                bop_v2.save_masks(
                    masks, v2_dir / (image_key + f".{mask_type}.json")
                )
        for im_modality in (
            "rgb",
            "gray",
            "depth",
        ):
            if scene_infos["has_" + im_modality]:
                im_path = list(
                    (v1_scene_dir / im_modality).glob(f"{image_id:06d}.*"))[0]
                suffix = im_path.suffix
                shutil.copy(
                    im_path,
                    v2_dir / (image_key + "." + im_modality + suffix)
                )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="BOP-V1 -> BOP-V2 converter utility",
    )
    parser.add_argument(
        "--input",
        help="""A directory containing the scenes of a dataset in v1
        format, e.g. ./ycbv/train_pbr.
        """,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="""Output directory that will contain the dataset
        in v2 format, e.g. ./ycbv/train_pbr_v2format.
        """,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scene-ids",
        help="""Scenes to be converted. Ids should be
        separated by commas, e.g. --scene-ids 4,5,6.
        If unspecified, all scenes will be converted.
        """,
        type=str,
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


def main(args):
    args = parse_args()

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output, exist_ok=True)

    if args.scene_ids is not None:
        scene_ids = [int(scene_id) for scene_id in args.scene_ids.split(",")]
        v1_scene_directories = (
            input_dir / f"{scene_id:06d}" for scene_id in scene_ids)
    else:
        v1_scene_directories = input_dir.iterdir()

    if args.nprocs > 0:
        v1_scene_directories = list(v1_scene_directories)
        _args = zip(
            v1_scene_directories,
            itertools.repeat(output_dir, len(v1_scene_directories))
        )
        with multiprocessing.Pool(processes=args.nprocs) as pool:
            with tqdm.tqdm(total=len(v1_scene_directories)) as pbar:
                iterator = pool.starmap(
                    convert_v1_scene_to_v2,
                    iterable=_args
                )
                for _ in iterator:
                    pbar.update()
    else:
        for v1_scene_directory in tqdm.tqdm(v1_scene_directories):
            convert_v1_scene_to_v2(
                v1_scene_directory,
                output_dir,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
