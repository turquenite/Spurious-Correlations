from enum import Enum

import numpy as np


class Position(Enum):
    LEFT_TOP = "left_top"
    LEFT_BOTTOM = "left_bottom"
    RIGHT_TOP = "right_top"
    RIGHT_BOTTOM = "right_bottom"


class Orientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def spurious_square(
    image: np.ndarray, pos: Position = Position.LEFT_TOP, size: int = 3
):
    _, height, width = image.shape
    match pos:
        case Position.LEFT_TOP:
            image[0, :size, :size] = 1

        case Position.LEFT_BOTTOM:
            image[0, height - size :, :size] = 1

        case Position.RIGHT_TOP:
            image[0, :size, width - size :] = 1

        case Position.RIGHT_BOTTOM:
            image[0, height - size :, width - size :] = 1

        case _:
            raise ValueError("Invalid Position.")

    return image


def spurious_lines(
    image: np.ndarray,
    orientation: Orientation = Orientation.VERTICAL,
    distance: int = 7,
):
    _, height, width = image.shape
    match orientation:
        case Orientation.VERTICAL:
            for x in range(0, width, distance):
                image[0, :, x] = 1

        case Orientation.HORIZONTAL:
            for y in range(0, height, distance):
                image[0, y, :] = 1

        case _:
            raise ValueError("Invalid Orientation")

    return image
