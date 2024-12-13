import math
import cv2 as cv
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike


def make_size_divisible_by(width: int, height: int, factor: int) -> Tuple[int, int]:
    rows = height // factor + (1 if height % factor else 0)
    cols = width // factor + (1 if width % factor else 0)   
    return cols * factor, rows * factor


def make_array_divisible_by(arr: ArrayLike, factor: int) -> np.ndarray:
    arr = np.asarray(arr)
    if not arr.ndim in [3, 4]:
        raise ValueError("Expected a 3D or 4D array as input")
    
    h, w = arr.shape[-3: -1]
    w_new, h_new = make_size_divisible_by(w, h, factor)
    pads = [(0, h_new - h), (0, w_new - w), (0, 0)]
    if arr.ndim == 4:
        pads.insert(0, (0, 0))

    return np.pad(arr, pads, mode='constant', constant_values=0)


def image_to_tiles(image: np.array, tile_size: int, tile_pad: int = 0) -> np.array:
    if not image.ndim in [2, 3]:
        raise ValueError("Expected a 2D or 3D array as input")

    h, w = image.shape[0:2] 

    # prepare expanded frame
    ts = tile_size - 2 * tile_pad
    assert ts > 0, "Tile padding is bigger than tile size itself"

    rows = h // ts + (1 if h % ts else 0)
    cols = w // ts + (1 if w % ts else 0)

    # pad image
    pads = [(tile_pad, rows * ts - h + tile_pad), (tile_pad, cols * ts - w + tile_pad)]
    if image.ndim == 3:
        pads += [(0, 0)]
    padded_image = np.pad(image, pads, mode='constant', constant_values=0)

    # split image to tiles
    tiles = []
    for y in range(0, rows * ts, ts):
        for x in range(0, cols * ts, ts):
            tile = padded_image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

    return np.stack(tiles, 0) # (n, h, w, c)

def tiles_to_image(tiles: np.array, frame_h: int, frame_w: int, tile_padding: int = 0):
    # Construct padded image
    n, h, w, c = tiles.shape
    pad = 2 * tile_padding
    ts_h = h - pad
    ts_w = w - pad
    rows = frame_h // ts_h + (1 if frame_h % ts_h else 0)
    cols = frame_w // ts_w + (1 if frame_w % ts_w else 0)
    padded_frame = np.zeros(shape=(rows * (h - pad) + pad, cols * (w - pad) + pad, c), dtype=tiles.dtype)

    tile_idx = 0
    for y in range(0, rows * (h - pad), (h - pad)):
        for x in range(0, cols * (w - pad), (w - pad)):
            padded_frame[y:y + h, x:x + w] = tiles[tile_idx]
            tile_idx += 1

    # Remove padding
    result_frame = padded_frame[tile_padding:tile_padding+frame_h, tile_padding:tile_padding+frame_w]

    return result_frame
