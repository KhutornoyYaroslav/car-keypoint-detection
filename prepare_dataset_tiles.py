import os
import json
import shutil
import logging
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from core.utils.string import str2bool
from core.utils.logger import setup_logger
from core.utils.dataset import parse_anno_object
from core.data.transforms.functional import image_to_tiles


_LOGGER_NAME = "DATASET PREP"


def prepare_dataset(src_path: str,
                    dst_root: str,
                    class_path: str,
                    tile_size: int,
                    point_radius: int,
                    val_perc: float = 0.2,
                    skip_zero: bool = False,
                    debug_show: bool = False,
                    filename_template: str = "%08d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    val_perc = np.clip(val_perc, 0.0, 1.0)

    # prepare output dirs
    dst_val_dir = os.path.join(dst_root, "val")
    shutil.rmtree(dst_val_dir, True)

    dst_train_dir = os.path.join(dst_root, "train")
    shutil.rmtree(dst_train_dir, True)

    # read label-index mapping
    with open(class_path, 'r') as f:
        class_map = json.load(f)

    # scan source annotation files
    src_annos = [f for f in sorted(glob(src_path)) if f.endswith("json")]

    # process sources
    saved_cnt = 0
    logger.info(f"Processing {len(src_annos)} source files ...")
    for src_anno in tqdm(src_annos):
        src_dir = os.path.dirname(os.path.abspath(src_anno))

        with open(src_anno, 'r') as f:
            anno = json.load(f)

        # read image
        src_img = None
        if "fname" in anno:
            img_path = os.path.join(src_dir, anno["fname"])
            if os.path.isfile(img_path):
                src_img = cv.imread(img_path, cv.IMREAD_COLOR)

        if src_img is None:
            img_path = os.path.splitext(src_anno)[0]
            if os.path.isfile(img_path):
                src_img = cv.imread(img_path, cv.IMREAD_COLOR)

        if (src_img is None) or isinstance(src_img, type(None)):
            logger.warning(f"Failed to find image for {src_anno}. Skip it.")
            continue

        # create label
        label = np.zeros(shape=src_img.shape[:2], dtype=np.uint8)

        # create roi
        roi = np.full(shape=src_img.shape[:2], fill_value=255, dtype=np.uint8)

        # process objects
        if "objects" not in anno:
            logger.warning(f"Failed to parse objects for {src_anno}. Skip it.")
            continue

        for obj in anno["objects"]:
            brect, keypoints = parse_anno_object(obj)
            for key, val in keypoints.items():
                if key in class_map:
                    cv.circle(label, val, point_radius, class_map[key], -1)

        # split to tiles
        img_tiles = image_to_tiles(src_img, tile_size)
        label_tiles = image_to_tiles(label, tile_size)
        roi_tiles = image_to_tiles(roi, tile_size)
        assert img_tiles.shape[:2] == label_tiles.shape[:2] == roi_tiles.shape[:2]

        for img_tile, label_tile, roi_tile in zip(img_tiles, label_tiles, roi_tiles):
            if skip_zero and np.count_nonzero(label_tile) == 0:
                continue

            # save to disk
            dst_subdir = dst_val_dir if np.random.choice(2, p=[1 - val_perc, val_perc]) else dst_train_dir
            dst_img_dir = os.path.join(dst_subdir, "img")
            os.makedirs(dst_img_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_img_dir, filename_template % saved_cnt), img_tile)
            dst_label_dir = os.path.join(dst_subdir, "label")
            os.makedirs(dst_label_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_label_dir, filename_template % saved_cnt), label_tile)
            dst_roi_dir = os.path.join(dst_subdir, "roi")
            os.makedirs(dst_roi_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_roi_dir, filename_template % saved_cnt), roi_tile)
            saved_cnt += 1

            # debug
            if debug_show:
                c = img_tile.shape[-1]
                debug_label = 7 * np.stack(c * [label_tile], -1)
                debug_roi = np.stack(c * [roi_tile], -1)
                debug_img = np.concatenate([img_tile, debug_label, debug_roi], axis=1)
                cv.imshow('img+label+roi', debug_img)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    return

    logger.info(f"Done. Saved {saved_cnt} tiles in total")


def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Dataset Preparing For Car Keypoint Detection In Traffic Surveillance Camera Images')
    parser.add_argument('--src-path', dest='src_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/sources/done_parts/*/*",
                        help="Pattern-like path to source dataset annotation files")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/512x512_tiles_r4",
                        help="Path where to save result dataset")
    parser.add_argument('--class-path', dest='class_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/512x512_tiles_r4/classes.json",
                        help="Path to file with label-index mapping of dataset classes")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=512,
                        help="Size of output tiles")
    parser.add_argument('--point-radius', dest='point_radius', type=int, default=4,
                        help="Radius in pixels of keypoints in label image")
    parser.add_argument('--val-perc', dest='val_perc', type=float, default=0.2,
                        help="Size of validation data as a percentage of total size")
    parser.add_argument('--skip-zero', dest='skip_zero', default=True, type=str2bool,
                        help="Whether to skip tiles without class indexes > 0")
    parser.add_argument('--debug-show', dest='debug_show', default=False, type=str2bool,
                        help="Whether to show resulting tiles or not")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # prepare dataset
    prepare_dataset(args.src_path,
                    args.dst_root,
                    args.class_path,
                    args.tile_size,
                    args.point_radius,
                    args.val_perc,
                    args.skip_zero,
                    args.debug_show)


if __name__ == "__main__":
    main()
