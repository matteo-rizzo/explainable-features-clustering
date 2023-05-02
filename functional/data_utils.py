import logging

from functional.arc_utils import make_divisible


def check_img_size(img_size: int, stride: int = 32, logger: logging.Logger = logging.getLogger(__name__)) -> int:
    # Verify img_size is a multiple of stride s
    new_size: int = make_divisible(img_size, int(stride))  # ceil gs-multiple
    if new_size != img_size:
        logger.warning(f'WARNING: --img-size {img_size:g} must be multiple '
                       f'of max stride {stride:g}, updating to {new_size:g}')
    return new_size
