import numpy as np


MAIN_RGB_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (153, 153, 255),
    (153, 51, 102),
    (255, 255, 204),
    (204, 255, 255),
    (102, 0, 102),
    (255, 128, 128),
    (0, 102, 204),
    (204, 204, 255),
    (0, 0, 128),
    (204, 255, 255),
    (0, 204, 255),
    (204, 255, 255),
    (204, 255, 204),
    (255, 255, 153),
    (153, 204, 255),
    (255, 153, 204),
    (204, 153, 255),
    (255, 204, 153)
]


def get_rgb_colors(num: int, mean = [0, 0, 0], scale = [1, 1, 1], zero_first: bool = True):
    result = []

    if num < 1:
        return result

    # special case for background class
    if zero_first:
        result += [(0, 0, 0)]

    # get main colors
    rest_len = num - len(result) 
    if rest_len > 0:
        result += MAIN_RGB_COLORS[:rest_len]

    # generate rest colors as random
    rest_len = num - len(result)
    if rest_len > 0:
        result += np.random.randint(0, 255, (rest_len, 3)).tolist()

    # normalize colors
    mean_ = 3 * mean if len(mean) == 1 else mean
    scale_ = 3 * scale if len(scale) == 1 else scale
    assert len(mean_) == 3 and len(scale_) == 3

    for i, c in enumerate(result):
        r = (c[0] - mean_[0]) / scale_[0]
        g = (c[1] - mean_[1]) / scale_[1]
        b = (c[2] - mean_[2]) / scale_[2]
        result[i] = (r, g, b)

    return result
