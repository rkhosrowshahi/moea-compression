import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of SalUn Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="./data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    ##################################### Fine-tuning setting #################################################
    parser.add_argument(
        "--steps", default=100, type=int, help="number of total steps to run EAs"
    )
    parser.add_argument(
        "--sid", default=55, type=int, help="Solution index in PF summary csv"
    )
    parser.add_argument(
        "--np", default=100, type=int, help="Number of population members"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Enable merge",
    )

    parser.add_argument(
        "--merge_method",
        type=str,
        default="center_to_end",
        help="Choose between: [center_to_end, left_to_right]",
    )

    return parser.parse_args()