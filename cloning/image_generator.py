import numpy as np
from functools import reduce
from typing import Callable, Generator, Tuple, List


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


def augmentations():
    return [random_vertical_flip()]


def transform_images(
        images: np.ndarray,
        angles: np.ndarray,
        transforms: List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray]]]
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

        yield (
            transform_images(image_batch, measurement_batch, augmentations())
            if augment
            else [image_batch, measurement_batch])

