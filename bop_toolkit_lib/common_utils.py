
import numpy as np


def adjust_img_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if np.max(img) <= 1:
        img = img * 255
    img = img.astype(np.uint8)
    return img

def cast_to_numpy(x, dtype=None) -> np.ndarray:
    if x is None or isinstance(x, str):
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array([cast_to_numpy(xx) for xx in x])
    elif isinstance(x, dict):
        return {k: cast_to_numpy(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        if dtype is not None:
            x = x.astype(dtype)
        return x
    elif isinstance(x, (int, float, complex, np.float32)):
        return x
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr
