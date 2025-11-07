
import traceback
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from unet import UNet
from losses_metrics import DiceLoss
from utils import SegmentationDataset


def run_full_train(root='./data', epochs=1, batch_size=2, device=None,
                    train_dir: str = None, val_dir: str = None, target_size: tuple = (256, 256),
                    criterion=None, save_dir: str = "./model_checkpoints",
                    history_path: str = "training_history.json"):
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Quick train on device: {device}")

    base = Path(root)

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
    
    

    # datasets & loaders (resize to model input_size to avoid spatial mismatches)
    
    train_ds = SegmentationDataset(str(train_imgs), str(train_masks), transform=None, target_size=target_size)
    val_ds = SegmentationDataset(str(val_imgs), str(val_masks), transform=None, target_size=target_size)

    # safe on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = UNet(device=device, input_size=target_size)
    criterion = criterion or DiceLoss()

    try:
        model.train_model(train_loader, val_loader, epochs=epochs, lr=1e-3,
                           criterion=criterion, save_dir=save_dir, 
                           save_history_path=history_path)
        print('training completed successfully.')
    except Exception:
        print('training failed with exception:')
        traceback.print_exc()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run full training on provided dataset.")
    parser.add_argument('--root', type=str, default='./data', help='Root directory for training data.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--train-dir', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--val-dir', type=str, required=True, help='Path to the validation data directory.')
    parser.add_argument('--target-size', type=int, nargs=2, default=(256, 256), help='Target size (height width) for images.')
    parser.add_argument('--save-dir', type=str, default='./model_checkpoints', help='Directory to save model checkpoints.') 
    args = parser.parse_args()

    run_full_train(root=args.root,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   device=args.device,
                   train_dir=args.train_dir,
                   val_dir=args.val_dir,
                   target_size=tuple(args.target_size))