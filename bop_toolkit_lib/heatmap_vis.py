import numpy as np


def combine_depth_diffs(masks, diffs, use_clip=False, clip_val=None):
    is_rgb = diffs[0].ndim==3
    combined = np.zeros_like(diffs[0], dtype=np.float32)
    imgs=[]
    for i, mask in enumerate(masks):
        # mask = d > combined
        # print(f"{i=} {mask.sum()}")
        a = diffs[i][mask]
        combined[mask] = a
        imgs.append(combined)
    if use_clip:
        clip_val = np.percentile(combined, 99.7) if clip_val is None else clip_val
        combined = np.clip(combined, 0, clip_val)
    return {
        "combined": combined,
        "imgs": imgs,
    }