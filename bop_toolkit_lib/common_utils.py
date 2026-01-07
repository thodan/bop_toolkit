import numpy as np
from PIL import Image


def adjust_img_for_plt(img):
    """Converts an image to the HWC uint8 format. For example, used for plt.imshow().

    :param img: Image in CHW or HWC format, numpy array or torch tensor. If a batch dimension exists, it must be 1.
    :return: Image in HWC uint8 format.
    """

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
    """Casts input to numpy array of a given type.
    
    :param x: Input data. Can be a numpy array, a torch tensor, a PIL image, a list/tuple/dict with the aforementioned types.
    :return: Numpy array or the same structure as input with numpy arrays.
    """

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
    elif isinstance(x, Image.Image):
        arr = np.array(x)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    elif isinstance(x, (int, float, complex, np.float32)):
        return x
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr
