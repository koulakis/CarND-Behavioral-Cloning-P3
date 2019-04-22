import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def augmentations() -> List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Image transforms performed in the train data in order to artificially generate new samples.

    Returns:
        a list of transforms
    """
    return [random_vertical_flip()]


def transform_images(
        images: np.ndarray,
        angles: np.ndarray,
        transforms: List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]
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


def load_image_registry(project_directory: str, image_set_directory: str) -> pd.DataFrame:
    """Loads the registry of a single image set.

    Args:
        project_directory: directory of the project
        image_set_directory: directory of the image set

    Returns:
        a data frame with the registry
    """
    return (
        pd.read_csv(
            f'./my-videos-center/{image_set_directory}/driving_log.csv',
            header=None,
            names=[
                'center_image', 'left_image', 'right_image',
                'steering_angle', 'throttle', 'break', 'speed'])
        .assign(
            center_image=lambda df:
            project_directory
                + f'/my-videos-center/{image_set_directory}/IMG'
                + df['center_image'].str.split('IMG').str[1])
        [['center_image', 'steering_angle']])


def load_image_registries(project_directory: str, image_set_directories: List[str]):
    """Loads the registry data frames for given image sets.

    Args:
        project_directory: directory of the project
        image_set_directories: directories of the image sets

    Returns:
        a data frame with all the registries concatenated
    """
    return (
        pd.concat(
            [load_image_registry(project_directory, directory)
             for directory in image_set_directories])
          .reset_index(drop=True))


def randomly_load_images(registries: pd.DataFrame, batch_size=32) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a batch of images randomly selected from the registries.

    Args:
        registries: a data frame with image absolute paths and corresponding angles
        batch_size: the size of a single batch

    Returns:
        a batch of images and angles
    """
    batch_idx = np.random.choice(registries.shape[0], batch_size)

    batch_registries = registries.iloc[batch_idx]

    images = np.stack(batch_registries['center_image'].map(plt.imread), axis=0)
    angles = batch_registries['steering_angle']

    return images, angles


def load_all_images(project_directory: str, image_set_directories: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all images and angles from given directories. To be used for faster training using in-memory data. Use
        only if the amount of data in the image sets can fit in memory. Alternatively use the
        'image_generator_from_files' which loads images to memory in batches during training.

    Args:
        project_directory: directory of the project
        image_set_directories: directories of the image sets

    Returns:
        all images and angles from the image sets
    """
    registries = load_image_registries(project_directory, image_set_directories)
    images = np.stack(registries['center_image'].map(plt.imread), axis=0)
    angles = registries['steering_angle']

    return images, angles


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


def image_generator_from_files(
        project_directory: str,
        image_set_directories: List[str],
        batch_size: int = 32,
        augment: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Loads the image registries for the given image sets and then uses them to randomly pick batches. If augmentation
        is on, then the augmentation random transforms are applied to the images.

    Args:
        project_directory: directory of the project
        image_set_directories: directories of the image sets
        batch_size: the size of each batch
        augment: toggles augmentation of the images

    Returns:
        a tuple of features and steering angles as two numpy arrays
    """
    registries = load_image_registries(project_directory, image_set_directories)

    while True:
        batch_images, batch_angles = randomly_load_images(registries, batch_size=batch_size)

        yield (
            transform_images(batch_images, batch_angles, augmentations())
            if augment
            else [batch_images, batch_angles])
