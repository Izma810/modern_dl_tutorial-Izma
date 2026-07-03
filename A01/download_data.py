from __future__ import annotations

import argparse

from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument("--data_dir", type=str, default="./data")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    datasets.MNIST(root=args.data_dir, train=True, download=True)
    datasets.MNIST(root=args.data_dir, train=False, download=True)
    print(f"MNIST downloaded to {args.data_dir}")


if __name__ == "__main__":
    main(parse_args())
