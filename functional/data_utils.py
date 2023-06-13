import logging
from typing import Dict, Tuple

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision.transforms import transforms as T

from functional.arc_utils import make_divisible


def check_img_size(img_size: int, stride: int = 32, logger: logging.Logger = logging.getLogger(__name__)) -> int:
    # Verify img_size is a multiple of stride s
    new_size: int = make_divisible(img_size, int(stride))  # ceil gs-multiple
    if new_size != img_size:
        logger.warning(f'WARNING: --img-size {img_size:g} must be multiple '
                       f'of max stride {stride:g}, updating to {new_size:g}')
    return new_size


def create_stratified_splits(dataset, n_splits=1, train_size=700, test_size=300):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=test_size)
    for i, (train_index, test_index) in enumerate(splitter.split(dataset.data, dataset.data._labels)):
        # FIXME: do yield
        return Subset(dataset, train_index), Subset(dataset, test_index)


def apply_with_p(transformation, parameters: Dict) -> T.RandomApply:
    *params, (_, p) = parameters.items()
    return T.RandomApply([transformation(**dict(params))], p=p)


def get_transform(params: Dict, img_size: Tuple):
    return T.Compose([
        T.Resize(img_size, antialias=True),
        T.RandomHorizontalFlip(**params["flip"]),
        # Rotates an image with random angle
        apply_with_p(T.RandomRotation, params["rotation"]),
        # Performs random affine transform on an image
        # apply_with_p(T.RandomAffine, params["random_affine"]),
        # Randomly transforms the morphology of objects in images and produces a see-through-water-like effect
        # apply_with_p(T.ElasticTransform, params["elastic_transform"]),
        # Crops an image at a random location
        T.Compose([apply_with_p(T.RandomCrop, params["random_crop"]),
                   T.Resize(img_size, antialias=True)]),
        # Randomly changes the brightness, saturation, and other properties of an image
        apply_with_p(T.ColorJitter, params["color_jitter"]),
        # Performs gaussian blur transform on an image
        apply_with_p(T.GaussianBlur, params["gaussian_blur"]),
        # Randomly selects a rectangle region in a torch Tensor image and erases its pixels (already has p)
        # T.RandomErasing(**params["random_erasing"]),
        # Performs random perspective transform on an image
        apply_with_p(T.RandomPerspective, params["random_perspective"]),
        T.ToTensor()
    ])
