from .car_dataset import CarDataset


def build_dataset(cfg, root_dir: str, is_train: bool):
    return CarDataset(cfg, root_dir, is_train)
