import os
import json
import shutil
import logging
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from tqdm import tqdm
from core.utils.string import str2bool
from core.utils.logger import setup_logger
from core.utils.dataset import parse_anno_object


_LOGGER_NAME = "DATASET PREP"


def create_multiclass_label(w: int, h: int, class_map: dict, keypoints: dict, radius: int):
    label = np.zeros(shape=(h, w), dtype=np.uint8)

    for key, pt in keypoints.items():
        if key in class_map:
            # for pt in pts:
            cv.circle(label, pt, radius, class_map[key], -1)

    return label


def is_point_in_brect(pt: tuple, brect: list):
    if (pt[0] < brect[0]) or (pt[0] >= brect[2]):
        return False
    if (pt[1] < brect[1]) or (pt[1] >= brect[3]):
        return False

    return True


def convert_dataset(src_path: str,
                    dst_root: str, 
                    class_path: str,
                    val_perc: float,
                    target_size: int,
                    radius: int,
                    debug_show: bool = False,
                    filename_template = "%08d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    val_perc = np.clip(val_perc, 0.0, 1.0)

    # scan annotation files
    src_annos = [f for f in sorted(glob(src_path)) if f.endswith("json")]

    # prepare output dirs
    dst_val_dir = os.path.join(dst_root, "val")
    shutil.rmtree(dst_val_dir, True)

    dst_train_dir = os.path.join(dst_root, "train")
    shutil.rmtree(dst_train_dir, True)

    # read label-index mapping
    with open(class_path, 'r') as f:
        class_map = json.load(f)

    # process files
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

        # process objects
        if "objects" not in anno:
            logger.warning(f"Failed to parse objects for {src_anno}. Skip it.")
            continue

        # # get all keypoints
        # keypoints_all = {}
        # for obj in anno["objects"]:
        #     brect, keypoints = parse_anno_object(obj)
        #     for key, val in keypoints.items():
        #         if key not in keypoints_all:
        #             keypoints_all[key] = [val]
        #         else:
        #             keypoints_all[key].append(val)
                    

        for obj in anno["objects"]:
            brect, keypoints = parse_anno_object(obj)

            if not len(brect):
                logger.warning(f"Failed to parse brect for {src_anno}. Skip it.")
                continue

            if not len(keypoints):
                logger.warning(f"Failed to parse keypoints for {src_anno}. Skip it.")
                continue

            brect_w = brect[2] - brect[0]
            brect_h = brect[3] - brect[1]
            if (brect_w <= 0) or (brect_h <= 0) or (brect_w > src_img.shape[1]) or (brect_h > src_img.shape[1]):
                logger.warning(f"Found invalid brect in {src_anno}. Skip it.")
                continue

            fx = target_size / brect_w
            fy = target_size / brect_h
            
            keypoints_in_brect = {}
            # for key, pts in keypoints_all.items():
            #     for pt in pts:
            #         if is_point_in_brect(pt, brect):
            #             x = int(fx * (pt[0] - brect[0]))
            #             y = int(fy * (pt[1] - brect[1]))
            #             if key not in keypoints_in_brect:
            #                 keypoints_in_brect[key] = [(x, y)]
            #             else:
            #                 keypoints_in_brect[key].append((x, y))
            for key, pt in keypoints.items():
                if is_point_in_brect(pt, brect):
                    x = int(fx * (pt[0] - brect[0]))
                    y = int(fy * (pt[1] - brect[1]))
                    keypoints_in_brect[key] = (x, y)

            # image
            car_img = src_img[brect[1]:brect[3], brect[0]:brect[2]]
            car_img = cv.resize(car_img, dsize=(target_size, target_size), interpolation=cv.INTER_CUBIC)

            # label
            car_label = create_multiclass_label(car_img.shape[1], car_img.shape[0], class_map, keypoints_in_brect, radius)

            # save to disk
            dst_subdir = dst_val_dir if np.random.choice(2, p=[1 - val_perc, val_perc]) else dst_train_dir

            dst_img_dir = os.path.join(dst_subdir, "img")
            os.makedirs(dst_img_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_img_dir, filename_template % saved_cnt), car_img)

            dst_label_dir = os.path.join(dst_subdir, "label")
            os.makedirs(dst_label_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_label_dir, filename_template % saved_cnt), car_label)

            saved_cnt += 1

            # debug
            if debug_show:
                c = car_img.shape[-1]
                debug_label = 7 * np.stack(c * [car_label], -1)
                debug_img = np.concatenate([car_img, debug_label], axis=1)
                cv.imshow('img+label', debug_img)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    return

    logger.info(f"Done. Saved {saved_cnt} samples")


def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Dataset Preparing For Car Keypoint Detection In Traffic Surveillance Camera Images')
    parser.add_argument('--src-path', dest='src_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/sources/kp_done_parts/*/*",
                        help="Pattern-like path to annotation files")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/256x256_r5_brect_new",
                        help="Path where to save result dataset")
    parser.add_argument('--class-path', dest='class_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/256x256_r5_brect_new/classes.json",
                        help="Path to file with label-index mapping of dataset classes")
    parser.add_argument('--target-size', dest='target_size', type=int, default=256,
                        help="Size (width, height) of result dataset images")
    parser.add_argument('--point-radius', dest='point_radius', type=int, default=5,
                        help="Radius in pixels of keypoints in label image")
    parser.add_argument('--val-perc', dest='val_perc', type=float, default=0.1,
                        help="Size of validation data as a percentage of total size")
    parser.add_argument('--debug-show', dest='debug_show', default=False, type=str2bool,
                        help="Whether to show results or not")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # calculate dataset stats
    convert_dataset(args.src_path,
                    args.dst_root,
                    args.class_path,
                    args.val_perc,
                    args.target_size,
                    args.point_radius,
                    args.debug_show)


if __name__ == "__main__":
    main()
