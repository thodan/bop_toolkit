import argparse
import itertools
import multiprocessing
import pathlib
import shutil

import tqdm

from bop_toolkit_lib.dataset import bop_imagewise, bop_scenewise


def convert_scene_to_imagewise(
    input_scene_dir,
    output_dir,
    image_tkey,
):
    """Converts a scene in bop-scenewise format to bop-imagewise.

    :param input_scene_dir:
    Directory containing observations and annotations in bop-scenewise format.
    :param output_dir: Directory where the data will be written
    in bop-imagewise format.
    :param image_tkey: Template path containing the string '{image_id}'.
    """
    scene_data = bop_scenewise.load_scene_data(
        input_scene_dir,
        load_scene_camera=True,
        load_scene_gt=True,
        load_scene_gt_info=True,
    )

    scene_infos = bop_scenewise.read_scene_infos(
        input_scene_dir,
    )

    output_dir.mkdir(exist_ok=True)

    bop_imagewise.save_scene_camera(
        scene_data["scene_camera"], output_dir / (image_tkey + ".camera.json")
    )

    bop_imagewise.save_scene_gt(
        scene_data["scene_gt"], output_dir / (image_tkey + ".gt.json")
    )

    bop_imagewise.save_scene_gt(
        scene_data["scene_gt_info"], output_dir / (image_tkey + ".gt_info.json")
    )

    image_ids = [int(k) for k in scene_data["scene_camera"].keys()]
    for image_id in image_ids:
        image_key = image_tkey.format(image_id=image_id)
        for mask_type in ("mask", "mask_visib"):
            if scene_infos["has_" + mask_type]:
                masks = bop_scenewise.load_masks(
                    input_scene_dir, image_id, mask_type=mask_type
                )
                bop_imagewise.save_masks(
                    masks, output_dir / (image_key + f".{mask_type}.json")
                )
        for im_modality in (
            "rgb",
            "gray",
            "depth",
        ):
            if scene_infos["has_" + im_modality]:
                im_path = list(
                    (input_scene_dir / im_modality).glob(f"{image_id:06d}.*")
                )[0]
                suffix = im_path.suffix
                shutil.copy(
                    im_path, output_dir / (image_key + "." + im_modality + suffix)
                )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="BOP-scenewise -> BOP-imagewise converter utility",
    )
    parser.add_argument(
        "--input",
        help="""A directory containing the scenes of a dataset in
        format, e.g. ./ycbv/train_pbr.
        """,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="""Output directory that will contain the dataset
        in imagewise format, e.g. ./ycbv/train_pbr_imwise.
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
        scene_directories = (input_dir / f"{scene_id:06d}" for scene_id in scene_ids)
    else:
        scene_directories = input_dir.iterdir()

    if args.nprocs > 0:
        scene_directories = list(scene_directories)
        image_tkeys = [
            f"{scene_id:06d}_" + "{image_id:06d}"
            for scene_id in [int(d.name) for d in scene_directories]
        ]

        _args = zip(
            scene_directories,
            itertools.repeat(output_dir, len(scene_directories)),
            image_tkeys,
        )
        with multiprocessing.Pool(processes=args.nprocs) as pool:
            with tqdm.tqdm(total=len(scene_directories)) as pbar:
                iterator = pool.starmap(convert_scene_to_imagewise, iterable=_args)
                for _ in iterator:
                    pbar.update()
    else:
        for scene_directory in tqdm.tqdm(scene_directories):
            convert_scene_to_imagewise(
                scene_directory,
                output_dir,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
