import math
import torch
import cv2 as cv
import numpy as np
from core.utils.ops import xywh2xyxy, xyxy2xywh
from core.data.transforms.functional import make_array_divisible_by
from typing import Dict, Tuple, Sequence, Any, Optional, Union, List


class BaseTransform:
    def __init__(self):
        pass

    def apply_img(self, img: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to images.

        Args:
            'img' (numpy.ndarray): Array of images with shape (T, H, W, C),
                where T is sequence length, H is image height, W is image width,
                C is number of image channels.
        
        Returns:
            (numpy.ndarray): Transformed array of images with shape (T, H, W, C).
        """
        pass

    def apply_bbox(self, bbox: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to bounding boxes.
        If some bounding box is out of image borders after transformation,
        fills this box with zeros.

        Args:
            'bbox' (numpy.ndarray): Array of bounding boxes with shape (T, N, 4),
                where T is sequence length, N is number of bounding boxes per image.
                Assumes box coordinates are normalized in range [0, 1) and have format
                'cxcywh'.

        Returns:
            (numpy.ndarray): Transformed array of bounding boxes with shape (T, N, 4).
        """
        pass

    def apply_cls(self, cls: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to class scores of bounding boxes.

        Args:
            'cls' (numpy.ndarray): Array of class scores with shape (T, N, num_classes),
                where T is sequence length, N is number of bounding boxes per image.
                Assumes class scores are normalized in range [0, 1].

        Returns:
            (numpy.ndarray): Transformed array of class scores with shape (T, N, num_classes).
        """
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformation to data.

        Args:
            'data' (Dict): A dictionary containing image data and annotations.
                May include:
                    'img' (numpy.ndarray): The input image.
                    'bbox' (numpy.ndarray): Bounding boxes.
                    'cls' (numpy.ndarray): Class scores.

        Returns:
            (Dict): Transformed data dictionary.
        """


class Compose(BaseTransform):
    def __init__(self, transforms: Sequence[BaseTransform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class CheckFormat(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("Expected 'img' as numpy ndarray")
        if not img.ndim == 4:
            raise ValueError("Expected 'img' with shape (T, H, W, C)")
        return None

    def apply_bbox(self, bbox):
        if not isinstance(bbox, np.ndarray):
            raise ValueError("Expected 'bbox' as numpy ndarray")
        if not bbox.ndim == 3 or not bbox.shape[-1] == 4:
            raise ValueError("Expected 'bbox' with shape (T, N, 4)")
        return None
    
    def apply_cls(self, cls):
        if not isinstance(cls, np.ndarray):
            raise ValueError("Expected 'cls' as numpy ndarray")
        if not cls.ndim == 3:
            raise ValueError("Expected 'cls' with shape (T, N, num_classes)")

    def __call__(self, data):
        if 'img' in data:
            self.apply_img(data['img'])
        if 'bbox' in data:
            self.apply_bbox(data['bbox'])
        if 'cls' in data:
            self.apply_cls(data['cls'])
        return data


class ConvertColor(BaseTransform):
    def __init__(self, src: str, dst: str):
        super().__init__()
        self._str_to_cvtype(src, dst)

    def _str_to_cvtype(self, src: str, dst: str):
        if src == 'BGR' and dst == 'HSV':
            self.cvt_cvtype = cv.COLOR_BGR2HSV
        elif src == 'RGB' and dst == 'HSV':
            self.cvt_cvtype = cv.COLOR_RGB2HSV
        elif src == 'HSV' and dst == 'BGR':
            self.cvt_cvtype = cv.COLOR_HSV2BGR
        elif src == 'HSV' and dst == "RGB":
            self.cvt_cvtype = cv.COLOR_HSV2RGB
        elif src == 'RGB' and dst == 'BGR':
            self.cvt_cvtype = cv.COLOR_RGB2BGR
        elif src == 'BGR' and dst == 'RGB':
            self.cvt_cvtype = cv.COLOR_BGR2RGB
        else:
            raise NotImplementedError

    def apply_img(self, img):
        for i, _ in enumerate(img):
            img[i] = cv.cvtColor(img[i], self.cvt_cvtype)
        return None

    def __call__(self, data):
        if 'img' in data:
            self.apply_img(data['img'])
        return data


class Resize(BaseTransform):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def apply_img(self, img):
        res = []
        for i, _ in enumerate(img):
            res.append(cv.resize(img[i], self.size, interpolation=cv.INTER_AREA))
        return np.stack(res, 0)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class PadResize(BaseTransform):
    def __init__(self, size: Tuple[int, int], border_value: int = 114):
        super().__init__()
        self.size = size
        self.border_value = border_value

    def _calc_pads(self, img_w: int, img_h: int) -> List[int]:
        source_aspect = img_w / img_h
        target_aspect = self.size[0] / self.size[1]
        # tblr
        pads = [0, 0, 0, 0]
        if source_aspect > target_aspect:
            # pad top and bottom
            new_height = int(np.round(img_w / target_aspect))
            pads[0] = (new_height - img_h) // 2
            pads[1] = new_height - img_h - pads[0]
        else:
            # pad left and right
            new_width = int(np.round(img_h * target_aspect))
            pads[2] = (new_width - img_w) // 2
            pads[3] = new_width - img_w - pads[2]
        return pads

    def apply_img(self, img):
        t, h, w, c = img.shape
        # pad
        pads = self._calc_pads(w, h)
        img = np.pad(img,
                     [(0, 0), (pads[0], pads[1]), (pads[2], pads[3]), (0, 0)],
                     mode='constant', constant_values=self.border_value)
        # resize
        res = np.zeros(shape=(t, *self.size[::-1], c), dtype=img.dtype)
        for i, _ in enumerate(res):
            res[i] = cv.resize(img[i], self.size, interpolation=cv.INTER_AREA)
        return res
    
    def apply_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        pads = self._calc_pads(img_w, img_h)
        new_w = np.sum([img_w, *pads[2:]])
        new_h = np.sum([img_h, *pads[:2]])
        # to xyxy, denormalize
        bbox = xywh2xyxy(bbox)
        bbox *= [img_w, img_h, img_w, img_h]
        # translate bboxes
        bbox[..., 0::2] += pads[2] # x offset
        bbox[..., 1::2] += pads[0] # y offset
        # clip
        bbox[..., 0::2] = np.clip(bbox[..., 0::2], 0, new_w)
        bbox[..., 1::2] = np.clip(bbox[..., 1::2], 0, new_h)
        # filter empty bboxes (filling by zeros)
        bbox = xyxy2xywh(bbox)
        mask = bbox[..., 2] * bbox[..., 3] > 0
        mask = np.expand_dims(mask, -1).repeat(4, -1).astype(np.int32)
        bbox *= mask
        # normalize
        bbox /= [new_w, new_h, new_w, new_h]
        return bbox
    
    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            data['img'] = self.apply_img(data['img'])
            if 'bbox' in data:
                data['bbox'] = self.apply_bbox(data['bbox'], w, h)
        return data


class MakeDivisibleBy(BaseTransform):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def apply_img(self, img):
        return make_array_divisible_by(img, self.factor)

    def apply_bbox(self, bbox: np.ndarray, w_scale: float, h_scale: float):
        bbox[..., ::2] = bbox[..., ::2] * w_scale
        bbox[..., 1::2] = bbox[..., 1::2] * h_scale

    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            data['img'] = self.apply_img(data['img'])
            h_new, w_new = data['img'].shape[1:3]
            if 'bbox' in data:
                self.apply_bbox(data['bbox'], w / w_new, h / h_new)
        return data


class ToFloat(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        return img.astype(np.float32)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Normalize(BaseTransform):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super().__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def apply_img(self, img):
        return (img - self.mean_rgb) / self.scale_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Denormalize(BaseTransform):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super().__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def apply_img(self, img):
        return img * self.scale_rgb + self.mean_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Clip(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 255.0):
        super().__init__()
        self.min = min
        self.max = max
        assert self.max >= self.min, "min must be >= max"

    def apply_img(self, img):
        return np.clip(img, self.min, self.max)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class ToTensor(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        res = torch.from_numpy(img).type(torch.float32)
        if res.ndim == 4:
            res = res.permute(0, 3, 1, 2)
        elif res.ndim == 3:
            res = res.permute(2, 0, 1)
        else:
            raise ValueError("Expected 3D or 4D array")
        return res

    def apply_bbox(self, bbox):
        return torch.from_numpy(bbox).type(torch.float32)
    
    def apply_cls(self, cls):
        return torch.from_numpy(cls).type(torch.float32)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        if 'bbox' in data:
            data['bbox'] = self.apply_bbox(data['bbox'])
        if 'cls' in data:
            data['cls'] = self.apply_cls(data['cls'])
        return data


class ToNumpy(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img: torch.Tensor) -> np.ndarray:
        if img.dim() == 4:
            res = img.permute(0, 2, 3, 1)
        elif img.dim() == 3:
            res = img.permute(1, 2, 0)
        else:
            raise ValueError("Expected 3D or 4D array")
        return res.cpu().numpy()

    def apply_bbox(self, bbox: torch.Tensor) -> np.ndarray:
        return bbox.cpu().numpy()
    
    def apply_cls(self, cls: torch.Tensor) -> np.ndarray:
        return cls.cpu().numpy()

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        if 'bbox' in data:
            data['bbox'] = self.apply_bbox(data['bbox'])
        if 'cls' in data:
            data['cls'] = self.apply_cls(data['cls'])
        return data


class RandomJpeg(BaseTransform):
    def __init__(self, min_quality: float = 0.6, probabilty: float = 0.5):
        super().__init__()
        self.prob = np.clip(probabilty, 0.0, 1.0)
        self.min_quality = np.clip(min_quality, 0.0, 1.0)

    def apply_img(self, img):
        quality = min(self.min_quality + np.random.random() * (1.0 - self.min_quality), 1.0)
        encode_params = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * quality)]
        for i, _ in enumerate(img):
            _, encimg = cv.imencode('.jpg', img[i], encode_params)
            img[i] = cv.imdecode(encimg, 1)

    def __call__(self, data):
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                self.apply_img(data['img'])
        return data


class RandomPerspective(BaseTransform):
    def __init__(self,
                 rotate: float = 0.0,
                 translate: float = 0.0,
                 scale: float = 0.0,
                 shear: float = 0.0,
                 perspective: float = 0.0,
                 border_value: int = 114):
        super().__init__()
        self.rotate = np.clip(rotate, 0.0, 360.0)
        self.translate = np.clip(translate, 0.0, 1.0)
        self.scale = np.clip(scale, 0.0, 0.9)
        self.shear = np.clip(shear, 0.0, 90.0)
        self.perspective = np.clip(perspective, 0.0, 0.001)
        self.border_value = border_value

    def _construct_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        # center
        mat_c = np.eye(3, dtype=np.float32)
        mat_c[0, 2] = -img_w / 2  # x translation (pixels)
        mat_c[1, 2] = -img_h / 2  # y translation (pixels)

        # perspective
        mat_p = np.eye(3, dtype=np.float32)
        mat_p[2, 0] = np.random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        mat_p[2, 1] = np.random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # rotation and scale
        mat_r = np.eye(3, dtype=np.float32)
        a = np.random.uniform(-self.rotate, self.rotate)
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        mat_r[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # shear
        mat_s = np.eye(3, dtype=np.float32)
        mat_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        mat_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # translation
        mat_t = np.eye(3, dtype=np.float32)
        mat_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_w  # x translation (pixels)
        mat_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_h  # y translation (pixels)

        return mat_t @ mat_s @ mat_r @ mat_p @ mat_c

    def _box_candidates(self,
                        bbox1: np.ndarray, # original, (4, N), 'xyxy'
                        bbox2: np.ndarray, # augmented, (4, N), 'xyxy'
                        wh_thr: float = 2,
                        ar_thr: float = 100,
                        area_thr: float = 0.1,
                        eps: float = 1e-16):
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        # aspect ratio
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        # candidates
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

    def apply_img(self, img: np.ndarray, trans_mat: np.ndarray):
        bval = img.shape[-1]*[self.border_value]
        if np.any(trans_mat != np.eye(3)):
            for i, _ in enumerate(img):
                if self.perspective:
                    img[i] = cv.warpPerspective(img[i], trans_mat, dsize=None, borderValue=bval)
                else:
                    img[i] = cv.warpAffine(img[i], trans_mat[:2], dsize=None, borderValue=bval)

    def apply_bbox(self, bbox: np.ndarray, w: int, h: int, trans_mat: np.ndarray) -> np.ndarray:
        t, n = bbox.shape[0:2]

        # to xyxy, denormalize
        bbox = xywh2xyxy(bbox)
        bbox *= [w, h, w, h]

        # as corner points x,y,1
        total_boxes = bbox.shape[0] * bbox.shape[1]
        xy = np.ones(shape=(4 * total_boxes, 3), dtype=bbox.dtype)
        xy[:, :2] = bbox.reshape(-1, 4)[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * total_boxes, 2) # x1y1, x2y2, x1y2, x2y1

        # transform
        xy = xy @ trans_mat.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(total_boxes, 8) # perspective rescale or affine

        # new bboxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bbox.dtype).reshape(4, total_boxes).T
        new_bbox = new_bbox.reshape(t, n, 4)

        # clip
        new_bbox[..., 0::2] = np.clip(new_bbox[..., 0::2], 0, w)
        new_bbox[..., 1::2] = np.clip(new_bbox[..., 1::2], 0, h)

        # filter bad bboxes (filling by zeros)
        new_bbox = new_bbox.reshape(-1, 4)
        mask = self._box_candidates(bbox.reshape(-1, 4).T, new_bbox.T)
        mask = np.expand_dims(mask, -1).repeat(4, -1).astype(np.int32)
        new_bbox *= mask
        new_bbox = new_bbox.reshape(t, n, 4)

        # normalize, to xywh
        new_bbox /= [w, h, w, h]
        return xyxy2xywh(new_bbox)

    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            mat = self._construct_matrix(w, h)
            self.apply_img(data['img'], mat)
            if 'bbox' in data:
                data['bbox'] = self.apply_bbox(data['bbox'], w, h, mat)
        return data






# import torch
# import cv2 as cv
# import numpy as np


# class TransformCompose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, input, target = None, roi = None):
#         for t in self.transforms:
#             input, target, roi = t(input, target, roi)

#         target = 0 if target is None else target
#         roi = 0 if roi is None else roi

#         return input, target, roi


# class ConvertFromInts:
#     def __call__(self, input, target = None, roi = None):
#         input = input.astype(np.float32)
#         if target is not None:
#             target = target.astype(np.float32)
#         if roi is not None:
#             roi = roi.astype(np.float32)

#         return input, target, roi


# class ConvertToInts:
#     def __call__(self, input, target = None, roi = None):
#         input = input.astype(np.uint8)
#         if target is not None:
#             target = target.astype(np.uint8)
#         if roi is not None:
#             roi = roi.astype(np.uint8)

#         return input, target, roi


# class Clip(object):
#     def __init__(self, min: float = 0.0, max: float = 255.0):
#         self.min = min
#         self.max = max
#         assert self.max >= self.min, "max must be >= min"

#     def __call__(self, input, target = None, roi = None):
#         input = np.clip(input, self.min, self.max)

#         return input, target, roi


# class Normalize(object):
#     def __init__(self, mean = [0], scale = [255]):
#         self.mean = mean
#         self.scale = scale

#     def __call__(self, input, target = None, roi = None):
#         input = (input.astype(np.float32) - self.mean) / self.scale

#         return input, target, roi

# class Denormalize(object):
#     def __init__(self, mean = [0], scale = [255]):
#         self.mean = mean
#         self.scale = scale

#     def __call__(self, input, target = None, roi = None):
#         input = self.scale * input + self.mean

#         return input, target, roi


# class ToTensor:
#     def __call__(self, input, target = None, roi = None):
#         # check channels: (H, W) to (H, W, 1)
#         if input.ndim == 2:
#             input = np.expand_dims(input, axis=-1)
#         if target is not None and target.ndim == 2:
#             target = np.expand_dims(target, axis=-1)
#         if roi is not None and roi.ndim == 2:
#             roi = np.expand_dims(roi, axis=-1)

#         # to tensor
#         input = torch.from_numpy(input.astype(np.float32)).permute(2, 0, 1)
#         if target is not None:
#             target = torch.from_numpy(target.astype(np.int64)).permute(2, 0, 1)
#         if roi is not None:
#             roi = torch.from_numpy(roi.astype(np.float32)).permute(2, 0, 1)
#             roi = roi / 255.0

#         return input, target, roi


# class FromTensor:
#     def __init__(self, dtype = np.float32):
#         self.dtype = dtype

#     def __call__(self, input, target = None, roi = None):
#         input = input.permute(1, 2, 0).cpu().numpy().astype(self.dtype)
#         if target is not None:
#             target = target.permute(1, 2, 0).cpu().numpy().astype(self.dtype)
#         if roi is not None:
#             roi = 255.0 * roi
#             roi = roi.permute(1, 2, 0).cpu().numpy().astype(self.dtype)

#         return input, target, roi


# class RandomRotate(object):
#     def __init__(self, angle_min: float = -45.0, angle_max: float = 45.0, probability: float = 0.5):
#         assert angle_max >= angle_min, "angle max must be >= angle min"
#         self.angle_min = angle_min
#         self.angle_max = angle_max
#         self.probability = np.clip(probability, 0.0, 1.0)

#     def __call__(self, input, target = None, roi = None):
#         if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
#             angle = np.random.uniform(self.angle_min, self.angle_max)
#             # input
#             center = input.shape[1] / 2, input.shape[0] / 2
#             rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
#             input = cv.warpAffine(input, rot_mat, input.shape[1::-1], flags=cv.INTER_CUBIC, borderValue=0)
#             # target
#             if target is not None:
#                 center = target.shape[1] / 2, target.shape[0] / 2
#                 rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
#                 target = cv.warpAffine(target, rot_mat, target.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=0)
#             # roi
#             if roi is not None:
#                 center = roi.shape[1] / 2, roi.shape[0] / 2
#                 rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
#                 roi = cv.warpAffine(roi, rot_mat, roi.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=0)          

#         return input, target, roi


# class RandomCrop(object):
#     def __init__(self, min_size: float, max_size: float, probability: float = 0.5, keep_aspect: bool = False):
#         self.min_size = np.clip(min_size, 0.0, 1.0)
#         self.max_size = np.clip(max_size, 0.0, 1.0)
#         self.probability = np.clip(probability, 0.0, 1.0)
#         self.keep_aspect = keep_aspect

#     def __call__(self, input, target = None, roi = None):
#         if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
#             # random size
#             w_norm, h_norm = np.random.uniform(self.min_size, self.max_size, 2)
#             if self.keep_aspect:
#                 h_norm = w_norm
#             x_norm = np.random.random() * (1 - w_norm)
#             y_norm = np.random.random() * (1 - h_norm)

#             # crop
#             h, w = input.shape[0:2]
#             input = input[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
#                           int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

#             if target is not None:
#                 h, w = target.shape[0:2]
#                 target = target[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
#                                 int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

#             if roi is not None:
#                 h, w = roi.shape[0:2]
#                 roi = roi[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
#                           int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

#         return input, target, roi


# class Resize(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, input, target = None, roi = None):
#         input = cv.resize(input, self.size, interpolation=cv.INTER_AREA)
#         if target is not None:
#             target = cv.resize(target, self.size, interpolation=cv.INTER_NEAREST)
#         if roi is not None:
#             roi = cv.resize(roi, self.size, interpolation=cv.INTER_NEAREST)

#         return input, target, roi


# class RandomMirror(object):
#     def __init__(self, horizont_prob: float = 0.5, probability: float = 0.5):
#             self.horizont_prob = np.clip(horizont_prob, 0.0, 1.0)
#             self.probability = np.clip(probability, 0.0, 1.0)

#     def __call__(self, input, target = None, roi = None):
#         if np.random.choice([0, 1], size=1, p=[1-self.probability, self.probability]):
#             if np.random.choice([0, 1], size=1, p=[1-self.horizont_prob, self.horizont_prob]):
#                 input = input[:, ::-1]
#                 if target is not None:
#                     target = target[:, ::-1]
#                 if roi is not None:
#                     roi = roi[:, ::-1]
#             else:
#                 input = input[::-1]
#                 if target is not None:
#                     target = target[::-1]
#                 if roi is not None:
#                     roi = roi[::-1]

#         return input, target, roi


# class ConvertColor(object):
#     def __init__(self, current: str, transform: str):
#         self.transform = transform
#         self.current = current

#     def __call__(self, input, target = None, roi = None):
#         if self.current == 'BGR' and self.transform == 'HSV':
#             input = cv.cvtColor(input, cv.COLOR_BGR2HSV)
#         elif self.current == 'BGR' and self.transform == 'GRAY':
#             input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
#         elif self.current == 'RGB' and self.transform == 'HSV':
#             input = cv.cvtColor(input, cv.COLOR_RGB2HSV)
#         elif self.current == 'BGR' and self.transform == 'RGB':
#             input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
#         elif self.current == 'HSV' and self.transform == 'BGR':
#             input = cv.cvtColor(input, cv.COLOR_HSV2BGR)
#         elif self.current == 'HSV' and self.transform == "RGB":
#             input = cv.cvtColor(input, cv.COLOR_HSV2RGB)
#         else:
#             raise NotImplementedError

#         return input, target, roi
    

# class RandomGamma(object):
#     def __init__(self, lower: float = 0.5, upper: float = 2.0, probability: float = 0.5):
#         self.lower = np.clip(lower, 0.0, None)
#         self.upper = np.clip(upper, 0.0, None)
#         assert self.upper >= self.lower, "contrast upper must be >= lower."
#         self.probability = np.clip(probability, 0.0, 1.0)

#     def __call__(self, input, target = None, roi = None):
#         assert input.dtype == np.float32, "image dtype must be float"
#         if np.random.choice([0, 1], size=1, p=[1-self.probability, self.probability]):
#             gamma = np.random.uniform(self.lower, self.upper)
#             # if np.mean(input) > 100:
#             input = pow(input / 255., gamma) * 255. # TODO: check it

#         return input, target, roi


# class RandomHue(object):
#     def __init__(self, delta: float = 30.0, probability: float = 0.5):
#         self.delta = np.clip(delta, 0.0, 360.0)
#         self.probability = np.clip(probability, 0.0, 1.0)

#     def __call__(self, input, target = None, roi = None):
#         if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
#             input = cv.cvtColor(input, cv.COLOR_RGB2HSV)
#             input[:, :, 0] += np.random.uniform(-self.delta, self.delta)
#             input[:, :, 0][input[:, :, 0] > 360.0] -= 360.0
#             input[:, :, 0][input[:, :, 0] < 0.0] += 360.0
#             input = cv.cvtColor(input, cv.COLOR_HSV2RGB)

#         return input, target, roi


# class RandomJpeg(object):
#     def __init__(self, min_quality:float=0.6, probability:float=0.5):
#         self.probability = np.clip(probability, 0.0, 1.0)
#         self.min_quality = np.clip(min_quality, 0.0, 1.0)

#     def __call__(self, input, target = None, roi = None):
#         if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
#             quality = min(self.min_quality + np.random.random() * (1.0 - self.min_quality), 1.0)
#             encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * quality)]
#             _, encimg = cv.imencode('.jpg', input, encode_param)
#             input = cv.imdecode(encimg, 1)

#         return input, target, roi
