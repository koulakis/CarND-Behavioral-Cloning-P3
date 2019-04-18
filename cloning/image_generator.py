import numpy as np
import cv2
from functools import reduce
from typing import Callable, Generator, Tuple, List


def resize_to_square() -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Resize a cropped image to become square.

    Returns:
         a transform which applies the resizing
    """
    def resize(images, angles):
        height = images.shape[2]

        return np.stack([cv2.resize(img, (height, height)) for img in images]), angles

    return resize


def random_vertical_flip() -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Flips the images with a probability of 0.5.

    Returns:
        a transform which performs the flips
    """

    def random_flip(images, angles):
        flip = np.random.randint(2, size=angles.shape[0]).astype(bool)
        flipped_images = np.where(flip.reshape(angles.shape[0], 1, 1, 1), np.flip(images, axis=2), images)
        flipped_angles = np.where(flip, - angles, angles)
        return flipped_images, flipped_angles

    return random_flip


def scale_images(
        coeff: float = 1 / 255.0,
        bias: float = 0.5
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Scales images to an interval by applying a linear transform. The default interval is [-0.5, 0.5].

    Args:
        coeff: scaling coefficient
        bias: scaling bias

    Returns:
        transform function which scales images
    """
    return lambda images, angles: (coeff * images - bias, angles)


def crop_images_vertically(
        top_crop: int = 60,
        bottom_crop: int = 20
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Crops an image vertically by removing given pixels from the top and bottom.

    Args:
        top_crop: pixels to remove from top of image
        bottom_crop: pixels to remove from bottom of image

    Return:
        transform function which applies the crop on images
    """
    return lambda images, angles: (images[:, top_crop:(images.shape[1] - bottom_crop), :, :], angles)


def transform_images(
        images: np.ndarray,
        angles: np.ndarray,
        transforms: List[Callable[[np.ndarray, np.ndarray], List[np.ndarray]]]
) -> List[np.ndarray]:
    """Applies a list of transforms to all the images and measurements of a given batch.

    Args:
        images: array of stacked images
        angles: angles for the corresponding images
        transforms: list of functions which transform an image

    Returns:
        a transformed block of images
    """
    return reduce(lambda acc, nxt: nxt(*acc), transforms, [images, angles])


def image_generator_from_dataset(
        images: np.array,
        angles: np.array,
        batch_size: int = 32,
        augment: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """This generator yields the next training batch
    Args:
        images: image cached data to transform and serve in batches
        angles: corresponding angles to the image data
        batch_size: number of images generated on each batch
        augment: augment images with transforms


    Returns:
        a tuple of features and steering angles as two numpy arrays
    """
    data_size = angles.shape[0]
    while True:
        sample_idx = np.random.choice(data_size, batch_size)
        image_batch = images[sample_idx]
        measurement_batch = angles[sample_idx]

        standard_transforms = [crop_images_vertically(), scale_images(), resize_to_square()]
        augmentations = [random_vertical_flip()] if augment else []

        yield transform_images(
            image_batch,
            measurement_batch,
            standard_transforms + augmentations)
