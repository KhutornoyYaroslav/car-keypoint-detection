import argparse
from core.config import cfg
from core.data.datasets import build_dataset
# from core.data.transforms import build_transforms


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Car Keypoint Detection In Traffic Surveillance Dataset Visualization')
    parser.add_argument("-d", "--data-root", dest="data_root", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/CAR_KEYPOINTS/sources/kp_done_parts/done_part14")
    parser.add_argument("-t", "--istrain", dest="istrain", required=False, type=str2bool,
                        default=True)
    parser.add_argument("-r", "--frame-rate", dest="frame_rate", required=False, type=int,
                        default=25)
    args = parser.parse_args()

    # set config
    cfg.INPUT.IMAGE_SIZE = [1920, 1080]
    cfg.DATASET.CLASS_LABELS = [
        'fr wheel',
        'br wheel',
        'rear window tl',
        'rear window tr',
        'rear window br',
        'rear window bl',
        'rearview mirror r',
        'rearview mirror l',
        'bottom of license bl',
        'bottom of license br',
        'headlight bl inner bottom',
        'headlight bl outer top',
        'headlight br inner bottom',
        'headlight br outer top',
        'bottom bumper bl',
        'bottom bumper br',
        'side window back r',
        'windshield tr',
        'windshield tl',
        'windshield bl',
        'windshield br',
        'bottom of license fr',
        'bottom of license fl',
        'headlight fr inner bottom',
        'headlight fr outer top',
        'headlight fl inner bottom',
        'headlight fl outer top',
        'bottom bumper fr',
        'bottom bumper fl',
        'side window back l',
        'fl wheel',
        'bl wheel'
    ]
    cfg.freeze()

    # check dataset
    # transforms = build_transforms(cfg, is_train=args.istrain)
    # dataset = build_dataset(cfg, args.data_path, args.anno_path, transforms)
    dataset = build_dataset(cfg, args.data_root, args.istrain)
    print(f"Dataset size: {len(dataset)}")
    dataset.visualize(args.frame_rate)


if __name__ == '__main__':
    main()
