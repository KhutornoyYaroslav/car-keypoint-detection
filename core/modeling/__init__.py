from torch import nn
from core.modeling.unet.res_unet_plus_plus import ResUnetPlusPlus
from core.modeling.yolov8 import build_yolov8pose


def build_model(cfg) -> nn.Module:
    arch = cfg.MODEL.ARCHITECTURE

    if arch == "ResUnetPlusPlus":
        return ResUnetPlusPlus(cfg)
    elif arch.startswith("yolov8"):
        return build_yolov8pose(cfg)
    else:
        raise ValueError(f"Model architecture '{arch}' not implemented")
