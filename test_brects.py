import os
import torch
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from core.config import cfg as cfg
from core.modeling import build_model
from core.utils.logger import setup_logger
from core.utils.colors import get_rgb_colors
from core.utils.tensorboard import draw_labels
from core.utils.checkpoint import CheckPointer
from core.data.transforms.functional import image_to_tiles, tiles_to_image
from ultralytics import YOLO
from typing import List
from core.data.transforms.transforms import (
    Clip,
    Resize,
    ToTensor,
    Normalize,
    ConvertColor,
    ConvertFromInts,
    TransformCompose
)
from core.utils.car_keypoints import car_edges, car_nodes


LOGGER_NAME = "MODEL TEST"
IMSHOW_MAX_SIZE = 1200


def create_keypoint_model(cfg):
    device = torch.device(cfg.MODEL.DEVICE)   

    model = build_model(cfg)
    model.to(device)
    model.eval()

    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    return model


def detect_cars(img: np.ndarray,
                yolo_model, 
                min_conf: float = 0.5,
                device: str = 'cuda',
                classes: List[int] = [2]) -> List[List[int]]:
    results = []

    if img is None or not img.size:
        return results

    predicts = yolo_model.predict(img, conf=min_conf, classes=classes, device=device) #, imgsz=640)

    for pred in predicts:
        confs = pred.boxes.conf.cpu().numpy()
        boxes = pred.boxes.xyxy.cpu().numpy().astype(int)

        for c, b in zip(confs, boxes):
            results.append(b.tolist())

    return results


def calc_point_positions(binmask: np.array):
    positions = []
    areas = []
    contours, h = cv.findContours(binmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        M = cv.moments(c)
        cX = int(M["m10"] / (M["m00"] + 1e-6))
        cY = int(M["m01"] / (M["m00"] + 1e-6))
        positions.append((cX, cY))
        areas.append(cv.contourArea(c))

    return positions, areas


def detect_keypoints(cfg, frame, keypoint_model, device, close_dist: float = 0.25):
    # pre-process
    transforms = [
        ConvertColor('BGR', 'RGB'),
        Resize(cfg.INPUT.IMAGE_SIZE),
        ConvertFromInts(),
        Clip(),
        Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToTensor()
    ]
    transforms = TransformCompose(transforms)

    input, _, _ = transforms(frame)
    input = input.unsqueeze(0).to(device)

    # predict
    output = keypoint_model(input)
    output = output.squeeze(0)
    pred_labels = torch.softmax(output, dim=0).argmax(dim=0) # (h, w)
    pred_labels = pred_labels.cpu().numpy()

    # post-process
    fx = frame.shape[1] / cfg.INPUT.IMAGE_SIZE[0]
    fy = frame.shape[0] / cfg.INPUT.IMAGE_SIZE[1]
    dist_thresh = close_dist * np.linalg.norm(cfg.INPUT.IMAGE_SIZE)

    # rgb_colors = get_rgb_colors(len(cfg.DATASET.CLASS_LABELS))
    result = {}
    for class_id, class_label in enumerate(cfg.DATASET.CLASS_LABELS):
        if class_id == 0:
            continue

        class_mask = np.array(pred_labels == class_id, dtype=np.uint8)
        points, mask_areas = calc_point_positions(class_mask)

        # scale to original size
        for i, _ in enumerate(points):
            points[i] = int(fx * points[i][0]), int(fy * points[i][1])

        # filter close points of the same class
        idxs_to_remove = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                pt1, pt2 = points[i], points[j]
                dist = np.linalg.norm([pt1[0] - pt2[0], pt1[1] - pt2[1]])
                if dist < dist_thresh:
                    idxs_to_remove.append(i if (mask_areas[i] < mask_areas[j]) else j)
        idxs_to_remove = set(idxs_to_remove)
        points = [p for i, p in enumerate(points) if i not in idxs_to_remove]
        # mask_areas = [m for i, m in enumerate(mask_areas) if i not in idxs_to_remove]

        # filter zero points # TODO: CNN bug ?
        points = [p for p in points if p[0] != 0 and p[1] != 0]

        # add to keypoints
        if class_label not in result:
            result[class_label] = points
        else:
            result[class_label].extend(points)

    return result


def draw_skeleton(frame, keypoints: dict):
    assert sorted(car_nodes) == sorted(list(keypoints.keys()))

    color = np.random.randint(100, 255, 3).tolist()
    # color = (0, 0, 255)
    mask = np.zeros_like(frame)

    for edge in car_edges:
        pt1_idx, pt2_idx = edge[0], edge[1]
        pt1_label, pt2_label = car_nodes[pt1_idx], car_nodes[pt2_idx]

        pts1 = keypoints[pt1_label]
        pts2 = keypoints[pt2_label]

        if len(pts1) == 1 and len(pts2) == 1:
            pt1 = pts1[0]
            pt2 = pts2[0]
            cv.line(mask, pt1, pt2, color, 1, 16)

    return mask




def test_model(cfg, video_path: str):
    # create models
    keypoint_model = create_keypoint_model(cfg)
    yolo_model = YOLO('yolov8m.pt', 'detect')

    # process video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # detect cars
        brects = detect_cars(frame, yolo_model, 0.5)


        mask = np.zeros_like(frame)

        # detect keypoints
        for br in brects:
            car_img = frame[br[1]:br[3], br[0]:br[2]]
            keypoints = detect_keypoints(cfg, car_img, keypoint_model, cfg.MODEL.DEVICE)
            mask[br[1]:br[3], br[0]:br[2]] += draw_skeleton(car_img, keypoints)

        frame = cv.addWeighted(frame, 0.3, mask, 0.7, 0.0)
            # for label, pts in keypoints.items():
            #     for pt in pts:
            #         cv.circle(car_img, pt, 2, (0, 0, 255), -1)

        # draw result
        # for br in brects:
        #     cv.rectangle(frame, br[0:2], br[2:4], (0, 200, 0), 1)

        # show result
        resize_k = IMSHOW_MAX_SIZE / np.max(frame.shape[0:2])
        frame_resized = cv.resize(frame, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_CUBIC)
        cv.imshow('input', frame_resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cloud Detection In Satellite Images Model Testing With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/huawei/sources/h265/mp4/train/test_19.mp4",
    #                     help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/huawei/sources/h264/mp4/road_test_33.mp4",
    #                     help='Path to source video to test')
    parser.add_argument('--video-path', dest="video_path", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/kolo2/kolo2.mkv",
                        help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/olvia_problem.mkv",
    #                     help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/carla/1920x1280_1.mp4",
    #                     help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/home/yaroslav/repos/vcm-ts/data/huawei/sources/yar.mkv",
    #                     help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/mp4/vecteezy_georgetown-city-malaysia-december-17-2023-top-view-of_38184906.mp4",
    #                     help='Path to source video to test')
    # parser.add_argument('--video-path', dest="video_path", required=False, type=str,
    #                     default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/mp4/test_9.mp4",
    #                     help='Path to source video to test')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # test model
    test_model(cfg, args.video_path)


if __name__ == "__main__":
    main()
