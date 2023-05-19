import pathlib
import numpy as np
import json
import time
import webdataset as wds

from bop_toolkit_lib.dataset import (
    bop_webdataset,
    bop_v2,
    bop_v1
)


def print_summary(timings):
    mean = np.mean(timings)
    std = np.std(timings)
    print(f"""{mean:.2f} ms ± {std:.2f} ms
min: {np.min(timings):.2f} ms, max: {np.max(timings):.2f} ms
({len(timings)} samples)
""")


def benchmark_wds_sequential(shard_path, n_images=1000):
    shard_list = [str(shard_path)]

    def sample_decoder(iterator):
        for sample in iterator:
            yield bop_webdataset.decode_sample(
                sample,
                decode_camera=True,
                decode_rgb=True,
                decode_gray=False,
                decode_depth=True,
                decode_gt=True,
                decode_gt_info=True,
                decode_mask_visib=True,
                decode_mask=True,
            )
        return sample

    datapipeline = wds.DataPipeline(
        wds.SimpleShardList(shard_list),
        wds.tarfile_to_samples(),
        sample_decoder,
    )
    timings = []
    start = time.time()
    for n, sample in enumerate(datapipeline):
        timings.append((time.time() - start) * 1000)
        if n > n_images:
            break
        start = time.time()
    return timings


def benchmark_wds_random(wds_dir, n_images=100):

    key_to_shard = json.loads((wds_dir / 'key_to_shard.json').read_text())
    keys = list(key_to_shard.keys())
    np.random.RandomState(0).shuffle(keys)
    keys = keys[:n_images]

    timings = []
    start = time.time()
    for key in keys:
        shard_id = key_to_shard[key]
        bop_webdataset.load_image_data(
            wds_dir / f'shard-{shard_id:06d}.tar',
            key,
            load_rgb=True,
            load_gray=False,
            load_depth=True,
            load_mask_visib=True,
            load_mask=True,
            load_gt=True,
            load_gt_info=True,
        )
        timings.append((time.time() - start) * 1000)
        start = time.time()
    return timings


def benchmark_v2(v2_dir, n_images=1000):
    v2_file_paths = v2_dir.glob('*')
    keys = set([p.name.split('.')[0] for p in v2_file_paths])
    keys = list(keys)
    np.random.RandomState(0).shuffle(keys)

    timings = []
    start = time.time()
    for key in keys[:n_images]:
        bop_v2.load_image_data(
            v2_dir,
            key,
            load_rgb=True,
            load_gray=False,
            load_depth=True,
            load_mask_visib=True,
            load_mask=True,
            load_gt=True,
            load_gt_info=False,
        )
        timings.append((time.time() - start) * 1000)
        start = time.time()
    return timings


def benchmark_v1(v1_scene_dir, n_images=100):
    scene_infos = bop_v1.read_scene_infos(
        v1_scene_dir, read_image_ids=True)
    image_ids = scene_infos['image_ids']
    np.random.RandomState(0).shuffle(image_ids)[:n_images]

    timings = []
    start = time.time()
    for image_id in image_ids:
        bop_v1.load_image_data(
            v1_scene_dir,
            image_id,
            load_rgb=True,
            load_gray=False,
            load_depth=True,
            load_mask_visib=True,
            load_mask=True,
            load_gt=True,
            load_gt_info=False,
        )
        timings.append((time.time() - start) * 1000)
        start = time.time()
    return timings


if __name__ == '__main__':
    ycbv_dir = pathlib.Path('/media/ylabbe/usb/datasets/bop_datasets/ycbv/')

    timings = benchmark_wds_sequential(
        ycbv_dir / 'train_pbr_wdsformat' / f'shard-{0:06d}.tar',
        n_images=1000
    )
    print("# WebDataset, Sequential access")
    print_summary(timings[5:])

    timings = benchmark_wds_random(
        ycbv_dir / 'train_pbr_wdsformat',
        n_images=100,
    )
    print("# WebDataset, Random access")
    print_summary(timings[5:])

    timings = benchmark_v2(
        ycbv_dir / 'train_pbr_v2format',
        n_images=1000,
    )
    print("# V2 format, Random access")
    print_summary(timings[5:])

    timings = benchmark_v1(
        ycbv_dir / 'train_pbr' / f'{0:06d}',
        n_images=100,
    )
    print("# V1 format, Random access")
    print_summary(timings[5:])
