from enum import Enum

import numpy as np


class Position(Enum):
    LEFT_TOP = "left_top"
    LEFT_BOTTOM = "left_bottom"
    RIGHT_TOP = "right_top"
    RIGHT_BOTTOM = "right_bottom"


def spurious_square(
    image: np.ndarray, pos: Position = Position.LEFT_TOP, size: int = 3
):
    _, height, width = image.shape
    match pos:
        case Position.LEFT_TOP:
            image[:size, :size] = 1

        case Position.LEFT_BOTTOM:
            image[height - size :, :size] = 1

        case Position.RIGHT_TOP:
            image[:size, width - size :] = 1

        case Position.RIGHT_BOTTOM:
            image[height - size :, width - size :] = 1

        case _:
            raise ValueError("Invalid Position.")

    return image
