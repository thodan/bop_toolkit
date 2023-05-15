import argparse
import shutil
import pathlib
from bop_toolkit_lib.dataset import bop_v1, bop_v2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        prog='BOP-V1 -> BOP-V2 converter utility',
    )
    parser.add_argument('--input',
                        help="""
                        A directory containing the scenes of a dataset in v1
                        format, e.g. ./ycbv/train_pbr.
                        """, type=str, required=True)
    parser.add_argument('--output',
                        help="""
                        Output directory that will contain the dataset
                        in v2 format, e.g. ./ycbv/train_pbr_v2format
                        """, type=str, required=True)
    parser.add_argument('--scene-ids',
                        help="""
                        Scenes to be converted. If unspecified,
                        all scenes will be converted.
                        """, nargs='+', type=int)
    args = parser.parse_args()
    return args


def main(args):
    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output, exist_ok=True)

    if args.scene_ids is not None:
        scene_directories = (
            input_dir / f'{scene_id:06d}' for scene_id in args.scene_ids)
    else:
        scene_directories = input_dir.iterdir()
    
    for scene_dir in tqdm(list(scene_directories)):
        scene_id = int(scene_dir.name)

        scene_data = bop_v1.load_scene_data(
            scene_dir,
            load_scene_camera=True,
            load_scene_gt=True,
            load_scene_gt_info=True
        )

        scene_infos = bop_v1.read_scene_infos(
            scene_dir,
        )

        output_dir.mkdir(exist_ok=True)

        image_tkey = f'{scene_id:06d}_' + '{image_id:06d}'

        bop_v2.save_scene_camera(
            scene_data['scene_camera'],
            output_dir / (image_tkey + '.camera.json')
        )

        bop_v2.save_scene_gt(
            scene_data['scene_gt'],
            output_dir / (image_tkey + '.gt.json')
        )

        bop_v2.save_scene_gt(
            scene_data['scene_gt_info'],
            output_dir / (image_tkey + '.gt_info.json')
        )

        image_ids = [int(k) for k in scene_data['scene_camera'].keys()]
        for image_id in image_ids:
            image_key = image_tkey.format(image_id=image_id)
            for mask_type in ('mask', 'mask_visib'):
                if scene_infos['has_' + mask_type]:
                    masks = bop_v1.load_masks(
                        scene_dir, image_id, mask_type=mask_type)
                    bop_v2.save_masks(
                        masks,
                        output_dir / (image_key + f'.{mask_type}.json')
                    )
        
            for im_modality in ('rgb', 'gray', 'depth', ):
                if scene_infos['has_' + im_modality]:
                    im_path = list(
                        (scene_dir / im_modality).glob(f'{image_id:06d}.*'))[0]
                    suffix = im_path.suffix
                    shutil.copy(
                        im_path,
                        output_dir / (image_key + '.' + im_modality + suffix)
                    )


if __name__ == '__main__':
    main()
