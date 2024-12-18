import os
import json
import cv2 as cv
import numpy as np
from glob import glob
from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform


class PoseDataset(Dataset):
    def __init__(self, cfg: CfgNode, root_dir: str, transforms: BaseTransform):
        self.root_dir = root_dir
        self.kpt_class_labels = cfg.DATASET.CLASS_LABELS
        self.imgs, self.annos = self._scan_files(root_dir)
        assert len(self.imgs) == len(self.annos)
        self.transforms = transforms
        self.num_classes = 1
        self.num_kpts = len(self.kpt_class_labels)
        self.max_labels = 128 # TODO: as PAD_LABELS_TO from cfg

    def __len__(self):
        return len(self.imgs)

    def _scan_files(self, root_dir: str):
        imgs = []
        annos = sorted(glob(os.path.join(root_dir, "*.json")))
        for anno in annos:
            with open(anno, 'r') as f:
                data = json.load(f)
                fname = data['fname']
                dname = os.path.dirname(anno)
                iname = os.path.join(dname, fname)
                imgs.append(iname)
        return imgs, annos

    def _parse_anno(self, path: str):
        results = []
        with open(path, 'r') as f:
            data = json.load(f)
            for obj in data['objects']:
                assert obj['class'] == 'car'
                bbox = []
                kpts = {}
                for shape in obj['shapes']:
                    pts = shape['points']
                    if shape['type'] == 'BoundingRect':
                        if pts['top-left'] != None and pts['bottom-right'] != None:
                            x = pts['top-left']['x']
                            y = pts['top-left']['y']
                            w = pts['bottom-right']['x'] - x
                            h = pts['bottom-right']['y'] - y
                            bbox = [x + w / 2, y + h / 2, w, h] # cxcywh
                    if shape['type'] == 'Keypoints':
                        for label in self.kpt_class_labels:
                            assert label in pts
                            if pts[label] != None and pts[label]['x'] != None and pts[label]['y'] != None:
                                kpts[label] = (pts[label]['x'], pts[label]['y'])
                            else:
                                kpts[label] = None
                if len(bbox) == 4 and len(kpts) and not all(v == None for v in kpts.values()):
                    results.append((bbox, kpts))
                else:
                    print(f"Found empty object in: {path}")
        return results

    def __getitem__(self, idx):
        # read image
        img = cv.imread(self.imgs[idx], cv.IMREAD_COLOR)

        # read objects
        assert self.num_classes == 1
        box = np.zeros(shape=(self.max_labels, 4), dtype=np.float32)
        cls = np.zeros(shape=(self.max_labels, self.num_classes), dtype=np.float32)
        kpts = np.zeros(shape=(self.max_labels, self.num_kpts, 3), dtype=np.float32) # x, y, visible

        # list of (bbox, kpts), bbox - cxcywh, kpts - dict
        objects = self._parse_anno(self.annos[idx])
        for i, obj in enumerate(objects):
            # class
            cls[i][0] = 1.0
            # bbox
            box[i] = np.asarray(obj[0])
            # kpts
            for pt_idx, v in enumerate(obj[1].values()):
                if v != None:
                    kpts[i][pt_idx][0:2] = v
                    kpts[i][pt_idx][2] = 1.0

        # normalize coordinates
        box[:, 0::2] /= img.shape[1]
        box[:, 1::2] /= img.shape[0]
        kpts[..., 0] /= img.shape[1]
        kpts[..., 1] /= img.shape[0]

        item = {}
        item['img'] = np.stack([img], 0)    # (1, H, W, C)
        item['bbox'] = np.stack([box], 0)   # (1, max_labels, 4)
        item['cls'] = np.stack([cls], 0)    # (1, max_labels, num_classes)
        item['kpts'] = np.stack([kpts], 0)  # (1, max_labels, num_kpts, 3)

        # apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            item = self.__getitem__(i)
            for img, box, cls, kpts in zip(item["img"], item["bbox"], item["cls"], item['kpts']):
                img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                img_w, img_h = img.shape[-2:-4:-1]

                for b, c, k in zip(box, cls, kpts):
                    b = b.cpu().numpy()

                    # skip empty
                    if b[2] * b[3] == 0:
                        continue

                    # draw bbox
                    b[::2] *= img_w
                    b[1::2] *= img_h
                    tl = b[:2] - (b[2:4] / 2)
                    br = b[:2] + (b[2:4] / 2)
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 2)

                    # draw kpts
                    k = k.cpu().numpy()
                    for (x, y, visible) in k:
                        if visible == 0: continue
                        cv.circle(img, (int(x * img_w), int(y * img_h)), 1, (255, 0, 0), 2, -1)

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
