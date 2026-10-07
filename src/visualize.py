"""Save a grid of (masked input / prediction / ground truth) for a few test
digits, using a trained checkpoint.

Usage:
    python src/visualize.py --checkpoint ./checkpoints/best.pt --n 8
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import MaskedMNIST
from model import InpaintingCNN


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InpaintingCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_set = MaskedMNIST(train=False, mask_size=args.mask_size, data_dir=args.data_dir)

    fig, axes = plt.subplots(3, args.n, figsize=(2 * args.n, 6))
    with torch.no_grad():
        for i in range(args.n):
            masked_image, image, mask = test_set[i]
            pred = model(masked_image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))

            axes[0, i].imshow(masked_image.squeeze(), cmap="gray")
            axes[1, i].imshow(pred.squeeze().cpu(), cmap="gray")
            axes[2, i].imshow(image.squeeze(), cmap="gray")
            for row in range(3):
                axes[row, i].axis("off")

    axes[0, 0].set_ylabel("masked input")
    axes[1, 0].set_ylabel("prediction")
    axes[2, 0].set_ylabel("ground truth")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize inpainting results")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--mask-size", type=int, default=10)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--output", type=str, default="./outputs/examples.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
