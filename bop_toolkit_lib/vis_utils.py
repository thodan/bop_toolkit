import copy
import cv2


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
