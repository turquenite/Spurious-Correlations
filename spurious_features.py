"""This file includes all functions used to introduce spurious features in the MNIST-number dataset."""

from enum import Enum

import torch


class Position(Enum):
    """Contains all possible positions for the spurious square function."""

    LEFT_TOP = "left_top"
    LEFT_BOTTOM = "left_bottom"
    RIGHT_TOP = "right_top"
    RIGHT_BOTTOM = "right_bottom"


class Orientation(Enum):
    """Contains all possible orientations for the spurious lines function."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def spurious_square(
    image: torch.tensor,
    pos: Position = Position.LEFT_TOP,
    size: int = 3,
    only_spurious: bool = False,
) -> torch.tensor:
    """Introduces a spurious square in a given image.

    Args:
        image (torch.tensor): Image in which spurious square should be embedded.
        pos (Position, optional): Position where the square should occur. Defaults to Position.LEFT_TOP.
        size (int, optional): Size of the square. Defaults to 3.
        only_spurious (bool, optional): If true, an image with only the spurious features is returned.

    Raises:
        ValueError: Given an an invalid Position.

    Returns:
        torch.tensor: Image with embedded spurious square.
    """
    if only_spurious:
        image = torch.zeros_like(image)

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
    image: torch.tensor,
    orientation: Orientation = Orientation.VERTICAL,
    distance: int = 7,
    only_spurious: bool = False,
) -> torch.tensor:
    """Introduces spurious lines in a given image.

    Args:
        image (torch.tensor): Image in which spurious square should be embedded.
        orientation (Orientation, optional): Orientation in which the spurious lines should occur (horizontal or vertical). Defaults to Orientation.VERTICAL.
        distance (int, optional): Distance between spurious lines. Defaults to 7.
        only_spurious (bool, optional): If true, an image with only the spurious features is returned.


    Raises:
        ValueError: Given an invalid orientation.

    Returns:
        torch.tensor: Image with embedded spurious lines.
    """
    if only_spurious:
        image = torch.zeros_like(image)

    _, height, width = image.shape

    digit_mask = (image > 0).int()

    match orientation:
        case Orientation.VERTICAL:
            for x in range(0, width, distance):
                image[0, :, x] = torch.where(
                    digit_mask[0, :, x] == 0, 1, image[0, :, x]
                )

        case Orientation.HORIZONTAL:
            for y in range(0, height, distance):
                image[0, y, :] = torch.where(
                    digit_mask[0, y, :] == 0, 1, image[0, y, :]
                )

        case _:
            raise ValueError("Invalid Orientation")

    return image


class SpuriousFeature:
    """Class for creating spurious features."""

    def __init__(self, function: callable, description: str, **kwargs):
        """Initialize a spurious feature with a given function.

        Args:
            function (callable): A function which introduces a spurious feature.
            description (str): Short description concerning the spurious feature.
            **kwargs: Are passed to the given spurious function.
        """
        self.function = lambda img: function(img, **kwargs)
        self.description = description

    def apply(self, img: torch.tensor) -> torch.tensor:
        """Apply the spurious function to a given image.

        Args:
            img (torch.tensor): Image to which the spurious feature should be applied.

        Returns:
            torch.tensor: Image with embedded spurious feature.
        """
        return self.function(img)
