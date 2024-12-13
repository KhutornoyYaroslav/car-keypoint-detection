import os
import json
import logging
import argparse
import cv2 as cv
from glob import glob
from tqdm import tqdm
from core.utils.logger import setup_logger
from core.utils.dataset import parse_anno_object


_LOGGER_NAME = "DATASET STAT"


def calc_dataset_stat(src_path: str,
                      stat_path: str,
                      class_path: str,
                      bg_class_label: str = 'background'):
    """
    Calculates car keypoints dataset statistic.
    Counts number of each keypoint class in dataset and
    generates finall dataset class map.

    Parameters:
        src_path : str
            Pattern-like path to source dataset annotation files.
        stat_path : str
            Path to file where to save dataset statistic.
        class_path : str
            Path to file where to save dataset class map.
        bg_class_label : str
            Label used for zero background class (default = 'background').
    """
    logger = logging.getLogger(_LOGGER_NAME)

    # scan annotation files
    src_annos = [f for f in sorted(glob(src_path)) if f.endswith("json")]

    # calc stat
    logger.info(f"Calculating dataset statistics for {len(src_annos)} source files ...")

    keypoints_stat = {}
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

        # parse objects
        for obj in anno.get("objects", []):
            _, keypoints = parse_anno_object(obj)
            for key in keypoints.keys():
                if key in keypoints_stat:
                    keypoints_stat[key] += 1
                else:
                    keypoints_stat[key] = 0

    # save stat
    if len(stat_path):
        logger.info(f"Saving statistic info to {stat_path}.txt ...")
        stat_dir = os.path.dirname(os.path.abspath(stat_path))
        os.makedirs(stat_dir, exist_ok=True)
        with open(stat_path + ".txt", 'w') as f:
            f.write("%30s%10s\n" % ("class label", "count"))
            f.write("%40s\n" % (40 * "_"))
            for key, val in keypoints_stat.items():
                f.write("%30s%10d\n" % (key, val))

    # create final class map
    non_empty_labels = [k for (k, v) in keypoints_stat.items() if v > 0]
    if bg_class_label in non_empty_labels:
        logger.error(f"Reserved background class '{bg_class_label}' label already used in dataset. Please change label.")
        return None

    class_map = {bg_class_label: 0}
    for idx, label in enumerate(non_empty_labels):
        class_map[label] = idx + 1

    # save class map
    if len(class_path):
        logger.info(f"Saving class map to {class_path}.json ...")
        class_dir = os.path.dirname(os.path.abspath(class_path))
        os.makedirs(class_dir, exist_ok=True)
        with open(class_path + ".json", 'w') as f:
            json.dump(class_map, f, indent=4)

    logger.info("Done.")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Dataset Statistic Calculation For Car Keypoint Detection In Traffic Surveillance')
    parser.add_argument('--src-path', dest='src_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/sources/kp_done_parts/*/*",
                        help="Pattern-like path to source dataset annotation files")
    parser.add_argument('--stat-path', dest='stat_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/256x256_r5_brect_new/stat",
                        help="Path to file where to save result statistic")
    parser.add_argument('--class-path', dest='class_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/outputs/256x256_r5_brect_new/classes",
                        help="Path to file where to save classes label-index mapping")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # calculate dataset stat
    calc_dataset_stat(args.src_path, args.stat_path, args.class_path)


if __name__ == "__main__":
    main()
