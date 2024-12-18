from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform
from .pose_dataset import PoseDataset


def build_dataset(cfg: CfgNode,
                  root_dir: str,
                  transforms: BaseTransform):
    return PoseDataset(cfg, root_dir, transforms)
