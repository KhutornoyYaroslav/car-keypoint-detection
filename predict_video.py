import torch
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from core.config import cfg
from core.utils.checkpoint import CheckPointer
from core.utils.ops import non_max_suppression
from core.modeling import build_model
from core.data.transforms import build_transforms
from core.utils.car_keypoints import car_edges, car_nodes


def draw_skeleton(frame, keypoints: dict):
    color = np.random.randint(100, 255, 3).tolist()
    for edge in car_edges:
        pt1_idx, pt2_idx = edge[0], edge[1]
        pt1_label, pt2_label = car_nodes[pt1_idx], car_nodes[pt2_idx]

        if pt1_label in keypoints:
            pt = keypoints[pt1_label]
            cv.circle(frame, pt, 1, color, 2, -1)

        if pt2_label in keypoints:
            pt = keypoints[pt2_label]
            cv.circle(frame, pt, 1, color, 2     , -1)
    
        if pt1_label in keypoints and pt2_label in keypoints:
                pt1 = keypoints[pt1_label]
                pt2 = keypoints[pt2_label]
                cv.line(frame, pt1, pt2, color, 1, cv.LINE_AA)


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection With PyTorch')
    parser.add_argument('-c', '--cfg', dest='config_file', required=False, type=str, metavar="FILE",
                        default="configs/cfg.yaml",
                        help="path to config file")
    parser.add_argument('-i', '--input', dest='input', required=False, type=str, metavar="FILE",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/*.mp4",
                        default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/kolo2/kolo2.mkv",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/*.*",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/mp4/vecteezy_third-ring-road-in-moscow_28261173.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/huawei/sources/h265/mp4/train/test*.mp4",
                        help="path to input image")
    args = parser.parse_args()

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_model(cfg)
    model = model.to(device)
    model.eval()

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    # transforms
    transforms = build_transforms(cfg, False)

    # processs input
    inputs = sorted(glob(args.input))
    for input in inputs:
        print(f"Processing {input} ...")

        cap = cv.VideoCapture(input)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return -1

        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            if image is None:
                print("Failed to read frame")
                break

            detects = []
            with torch.no_grad():
                data = {
                    'img': np.stack([image], 0) # .copy() # (1, H, W, C)
                }
                np_keyframe = data['img'][-1, :, :, :].copy()

                data = transforms(data) # (1, C, H, W)

                keyframe = data['img']
                INPUT_IMG_SIZE = data['img'].shape[-1:-3:-1]

                out_y, _ = model(keyframe.to(device)) # (1, 4 + 1 + 32 * 3)

                # out_y = out_y[:, :5, :] # TODO: get bbox and cls only

                # out_y = out_y.permute(0, 2, 1)
                # outputs = Detect.postprocess(out_y, max_det=300, nc=cfg.MODEL.HEAD.NUM_CLASSES)
                # outputs = outputs.squeeze(0)

                # TODP: multi_label=True ?
                # outputs = non_max_suppression(out_y, conf_thres=0.0005, iou_thres=0.45, nc=cfg.MODEL.HEAD.NUM_CLASSES)[0]
                outputs = non_max_suppression(out_y, conf_thres=0.005, iou_thres=0.45, nc=1)[0]
                detects = outputs.to('cpu').numpy()

            for det in detects:
                x, y, w, h, conf, class_idx = det[:6]
                kpts = det[6:]
                kpts = np.asanyarray(kpts).reshape(-1, 3)

                img_h, img_w  = np_keyframe.shape[:2]
                x = (x / INPUT_IMG_SIZE[0]) * img_w
                y = (y / INPUT_IMG_SIZE[1]) * img_h
                w = (w / INPUT_IMG_SIZE[0]) * img_w
                h = (h / INPUT_IMG_SIZE[1]) * img_h
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                if conf > 0.5:
                    # color = (0, 255, 0)
                    # cv.rectangle(np_keyframe, (x, y), (w, h), color, 2) # NMS
                    # text = f"{int(class_idx)} {conf:.2f}"
                    # cv.putText(np_keyframe, text, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # cfg.DATASET.CLASS_LABELS
                    keypoints = {}
                    for i, pt in enumerate(kpts):
                        pt_x, pt_y, vis = pt
                        pt_x = (pt_x / INPUT_IMG_SIZE[0]) * img_w
                        pt_y = (pt_y / INPUT_IMG_SIZE[1]) * img_h

                        label = cfg.DATASET.CLASS_LABELS[i]
                        if vis > 0.5:
                            keypoints[label] = (int(pt_x), int(pt_y))
                            # cv.circle(np_keyframe, (int(pt_x), int(pt_y)), 1, (0, 0, 255), 2, -1)
                    draw_skeleton(np_keyframe, keypoints)

            resize_k = 1200.0 / np_keyframe.shape[1]
            cv.imshow('Result', cv.resize(np_keyframe, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_AREA))
            key = cv.waitKey(0) & 0xFF
            if key == ord(' '):
                break
            if key == ord('q'):
                return

    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
