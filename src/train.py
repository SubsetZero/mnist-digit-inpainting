"""Train the inpainting model on masked MNIST.

Usage:
    python src/train.py --epochs 20 --mask-size 10 --batch-size 128
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import InpaintingCNN


def masked_mse_loss(pred, target, mask, mask_weight=5.0):
    full = nn.functional.mse_loss(pred, target)
    if mask.sum() == 0:
        return full
    masked = nn.functional.mse_loss(pred * mask, target * mask, reduction="sum") / mask.sum()
    return full + mask_weight * masked


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for masked_image, image, mask in loader:
            masked_image, image, mask = masked_image.to(device), image.to(device), mask.to(device)
            pred = model(masked_image, mask)
            loss = masked_mse_loss(pred, image, mask)
            total_loss += loss.item() * masked_image.size(0)
            n += masked_image.size(0)
    return total_loss / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, mask_size=args.mask_size, data_dir=args.data_dir
    )

    model = InpaintingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    patience, wait = 5, 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for masked_image, image, mask in train_loader:
            masked_image, image, mask = masked_image.to(device), image.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(masked_image, mask)
            loss = masked_mse_loss(pred, image, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * masked_image.size(0)
            n += masked_image.size(0)

        train_loss /= n
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MNIST digit-inpainting model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-size", type=int, default=10, help="base size of the square mask")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
