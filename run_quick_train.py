import os
import shutil
import random
import traceback
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader

from unet import UNet
from losses_metrics import DiceLoss
from utils import *


def generate_synthetic_pairs(out_images_dir, out_masks_dir, n=4, size=(256, 256)):
    """Generate n synthetic image/mask pairs: white rectangles on black background."""
    for i in range(n):
        img = Image.new('RGB', size, color=(0, 0, 0))
        mask = Image.new('L', size, color=0)
        draw = ImageDraw.Draw(img)
        mdraw = ImageDraw.Draw(mask)

        # random rectangle
        w, h = size
        x0 = random.randint(10, w // 3)
        y0 = random.randint(10, h // 3)
        x1 = random.randint(w // 2, w - 10)
        y1 = random.randint(h // 2, h - 10)
        color = tuple([random.randint(100, 255) for _ in range(3)])
        draw.rectangle([x0, y0, x1, y1], fill=color)
        mdraw.rectangle([x0, y0, x1, y1], fill=255)

        img.save(out_images_dir / f"img_{i:03d}.png")
        mask.save(out_masks_dir / f"img_{i:03d}.png")


def run_quick_train(root='./data/quick_test', epochs=1, batch_size=2, device=None, use_real: bool = False,
                    train_dir: str = None, val_dir: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Quick train on device: {device}")

    base = Path(root)

    if use_real:
        # use user-provided directories
        if not train_dir or not val_dir:
            raise ValueError("When use_real=True you must provide --train-dir and --val-dir paths.")
        train_imgs = Path(train_dir) / 'images'
        train_masks = Path(train_dir) / 'masks'
        val_imgs = Path(val_dir) / 'images'
        val_masks = Path(val_dir) / 'masks'
        if not train_imgs.exists() or not train_masks.exists():
            raise FileNotFoundError(f"Train dirs not found: {train_imgs}, {train_masks}")
        if not val_imgs.exists() or not val_masks.exists():
            raise FileNotFoundError(f"Val dirs not found: {val_imgs}, {val_masks}")
    else:
        # cleanup if exists
        if base.exists():
            shutil.rmtree(base)

        train_base = base / 'train'
        val_base = base / 'val'

        train_imgs, train_masks = make_dirs(train_base)
        val_imgs, val_masks = make_dirs(val_base)

        # generate small train/val synthetic data
    
        generate_synthetic_pairs(train_imgs, train_masks, n=4, size=(256, 256))
        generate_synthetic_pairs(val_imgs, val_masks, n=2, size=(256, 256))

    # datasets & loaders (resize to model input_size to avoid spatial mismatches)
    target_size = (256, 256)
    train_ds = SegmentationDataset(str(train_imgs), str(train_masks), transform=None, target_size=target_size)
    val_ds = SegmentationDataset(str(val_imgs), str(val_masks), transform=None, target_size=target_size)

    # safe on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = UNet(device=device, input_size=(256, 256))
    criterion = DiceLoss()

    try:
        model.train_model(train_loader, val_loader, epochs=epochs, lr=1e-3, criterion=criterion)
        print('Quick training completed successfully.')
    except Exception:
        print('Quick training failed with exception:')
        traceback.print_exc()
    
    


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='./quck_data')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default=None)
    p.add_argument('--use-real', action='store_true', help='Use real dataset provided by --train-dir/--val-dir')
    p.add_argument('--train-dir', default=None, help='Path to train folder containing images/ and masks/')
    p.add_argument('--val-dir', default=None, help='Path to val folder containing images/ and masks/')
    args = p.parse_args()

    run_quick_train(root=args.root, epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, use_real=True, train_dir=args.train_dir, val_dir=args.val_dir)


if __name__ == '__main__':
    _cli()
