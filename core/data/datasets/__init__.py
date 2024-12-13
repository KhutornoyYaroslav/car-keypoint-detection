# from .car_dataset import CarDataset
from .pose_dataset import PoseDataset


def build_dataset(cfg, root_dir: str, is_train: bool):
    return PoseDataset(cfg, root_dir, is_train)
