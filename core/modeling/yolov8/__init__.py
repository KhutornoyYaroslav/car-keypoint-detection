import torch
import logging
from torch import nn
from core.config import CfgNode
from core.modeling.yolov8.head import build_head
from core.modeling.yolov8.backbone import build_backbone
from typing import List, Any, Tuple


class YoloV8Pose(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> Any:
        f = self.backbone(x)
        y = self.head(list(f))

        return y

    def get_num_classes(self) -> int:
        return self.head.nc
    
    def get_strides(self) -> List[float]:
        return self.head.stride
    
    def get_dfl_num_bins(self) -> int:
        return self.head.dfl_bins
    
    def get_kpts_shape(self) -> Tuple[int, int]:
        return self.head.kpt_shape


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


def build_yolov8pose(cfg: CfgNode) -> nn.Module:
    logger = logging.getLogger('CORE')

    # build backbone
    backbone = build_backbone(cfg)

    # evaluate channels and strides
    channels2d = []
    strides = []
    x = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32) # (b, c, h, w)
    features2d = backbone(x)
    for f in features2d:
        channels2d.append(f.shape[1])
        strides.append(x.shape[-2] / f.shape[-2])

    # build head
    head = build_head(cfg, channels=channels2d, strides=strides)

    # build model
    model = YoloV8Pose(backbone, head)
    initialize_weights(model)

    # model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built. Parameters in total: {total_params}")

    return model
