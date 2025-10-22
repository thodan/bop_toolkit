import copy

import cv2
import numpy as np


def calc_mask_visib_percent(mask_visib, valid_mask):
    total_pixels = np.sum(valid_mask)
    visible_pixels = np.sum(mask_visib)
    percent_visible = (visible_pixels / total_pixels) * 100
    return percent_visible
    

def draw_text_in_ul(
    rgb, extra_text, size=1, thickness=3, start_pos=(10, 30), color=(255, 0, 0)
):
    rgb = cv2.putText(
        copy.deepcopy(rgb),
        extra_text,
        start_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return rgb
