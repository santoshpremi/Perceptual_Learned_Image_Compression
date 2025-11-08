import argparse
import torch


PROJECT_DATA_ROOT = "data"


def train_options():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="hflic_experiment",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=PROJECT_DATA_ROOT,
        type=str,
        required=False,
        help="(Deprecated) Base dataset directory"
    )
    parser.add_argument(
        "--train-root",
        default=f"{PROJECT_DATA_ROOT}/vimeo_test_clean",
        type=str,
        help="Root directory containing vimeo_test_clean (sequences/ + sep_testlist.txt)"
    )
    parser.add_argument(
        "--train-list",
        default="sep_testlist.txt",
        type=str,
        help="Relative or absolute path to the Vimeo sequence list file"
    )
    parser.add_argument(
        "--train-frame-index",
        default=4,
        type=int,
        help="Frame index (1-7) to use from each Vimeo sequence"
    )
    parser.add_argument(
        "--eval-root",
        default=PROJECT_DATA_ROOT,
        type=str,
        help="Root directory containing evaluation images"
    )
    parser.add_argument(
        "--eval-split",
        default="kodak",
        type=str,
        help="Subdirectory name for evaluation images (passed to ImageFolder)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_D",
        "--learning-rate-disc",
        dest="lr_D",
        default=1e-4,
        type=float,
        help="Learning rate for discriminator (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0483,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        nargs="+",
        default=[0],
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Enable CUDA"
    )
    parser.add_argument(
        "--no-cuda",
        dest="cuda",
        action="store_false",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=192,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    args = parser.parse_args()
    return args


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="hflic_test",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=f"{PROJECT_DATA_ROOT}/kodak",
        type=str,
        required=False,
        help="Evaluation dataset root"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Enable CUDA"
    )
    parser.add_argument(
        "--no-cuda",
        dest="cuda",
        action="store_false",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="kodak",
        help="split dataset for (default: %(default)s)",
    )
    parser.add_argument(
        "--list-file",
        type=str,
        default=None,
        help="Optional list file for sequence-based datasets (e.g. sep_testlist.txt)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=4,
        help="Frame index to use for sequence datasets (default: %(default)s)",
    )
    args = parser.parse_args()
    return args
