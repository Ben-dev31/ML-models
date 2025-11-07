
import os
from typing import Optional, Tuple
from unet import UNet, SegmentationDataset
from losses_metrics import DiceLoss, dice_coeff, iou_score
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms


class PredictionDataset(torch.utils.data.Dataset):
    """Loads images from a directory and returns (tensor, filename)."""
    def __init__(self, images_dir: str, transform=None, target_size: Optional[Tuple[int,int]] = None):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
        if transform:
            self.transform = transform
        else:
            if target_size:
                self.transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
            else:
                self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        return img_t, fname

def validate_prediction(ground_truth_dir: str, predicted_masks_dir: str):
    """Compare predicted mask files to ground-truth masks in a folder-by-folder manner.

    Both folders should contain mask images with matching filenames (or at least matching stems).
    Returns a dict with average 'dice' and 'iou'.
    """
    gt_dir = Path(ground_truth_dir)
    pred_dir = Path(predicted_masks_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predicted masks directory not found: {pred_dir}")

    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if len(pred_files) == 0:
        raise ValueError(f"No predicted mask files found in {pred_dir}")

    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for p in pred_files:
        stem = p.stem
        # find gt mask by same stem
        gt_file = None
        for g in gt_dir.iterdir():
            if g.is_file() and g.stem == stem:
                gt_file = g
                break
        if gt_file is None:
            # skip if no matching gt
            print(f"Warning: ground-truth mask for {p.name} not found in {gt_dir}, skipping")
            continue

        pred_img = Image.open(p).convert('L')
        gt_img = Image.open(gt_file).convert('L')

        # resize gt to pred size if needed
        if pred_img.size != gt_img.size:
            gt_img = gt_img.resize(pred_img.size, resample=Image.NEAREST)

        pred_t = transforms.ToTensor()(pred_img)  # [1,H,W]
        gt_t = transforms.ToTensor()(gt_img)

        # ensure 0..1 floats
        # compute metrics (wrap in batch dim)
        dice = float(dice_coeff(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).item())
        iou = float(iou_score(pred_t.unsqueeze(0), gt_t.unsqueeze(0), thr=0.5).item())

        total_dice += dice
        total_iou += iou
        n += 1

    if n == 0:
        raise ValueError("No mask pairs found to evaluate.")

    avg = {'dice': total_dice / n, 'iou': total_iou / n, 'n': n}
    print(f"Validation over {n} masks: Dice={avg['dice']:.4f}, IoU={avg['iou']:.4f}")
    return avg

def run_prediction(model_path: str, images_dir: str, out_masks_dir: str, device=None, batch_size=2, target_size: Optional[Tuple[int,int]] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running prediction on device: {device}")

    # build model
    model = UNet(in_channels=3, out_channels=1)

    # load checkpoint (support dict with 'model_state' or plain state_dict)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    
    # determine target size from model (if set) to ensure spatial compatibility
    model_target = getattr(model, 'input_size', None)
    if model_target is None:
        model_target = target_size
    dataset = PredictionDataset(images_dir, transform=None, target_size=model_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    out_masks_dir = Path(out_masks_dir)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (images, img_names) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for j in range(preds.shape[0]):
                pred_mask = preds[j, 0].cpu().numpy() * 255
                pred_img = Image.fromarray(pred_mask.astype(np.uint8))
                pred_img.save(out_masks_dir / img_names[j])

    print(f"Prediction completed. Masks saved to {out_masks_dir}")

    # Evaluate if ground-truth masks are available
    gt_masks_dir = Path(images_dir).parent / 'masks'
    if gt_masks_dir.exists():
        print("Ground-truth masks directory found, running evaluation...")
        evaluate_model(model_path, images_dir, str(gt_masks_dir), device=device, batch_size=batch_size, target_size=model_target)
    

def evaluate_model(model_path: str, images_dir: str, masks_dir: str, device=None, batch_size: int = 4, threshold: float = 0.5, target_size: Optional[Tuple[int,int]] = None):
    """Evaluate model predictions against ground-truth masks.

    Returns average Dice and IoU across dataset.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model on device: {device}")

    # load model
    model = UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # determine target size from model if not provided
    model_target = getattr(model, 'input_size', None)
    if model_target is None:
        model_target = target_size

    dataset = PredictionDataset(images_dir, transform=None, target_size=model_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    masks_path = Path(masks_dir)
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    total_dice = 0.0
    total_iou = 0.0
    n = 0

    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # build target batch
            b = probs.shape[0]
            targets = []
            for name in names:
                stem = Path(name).stem
                # try to find mask file by stem
                mask_file = None
                for p in masks_path.iterdir():
                    if p.is_file() and p.stem == stem:
                        mask_file = p
                        break
                if mask_file is None:
                    raise FileNotFoundError(f"Ground-truth mask for {name} not found in {masks_dir}")
                m = Image.open(mask_file).convert('L')
                if model_target:
                    m = m.resize(model_target, resample=Image.NEAREST)
                m_t = transforms.ToTensor()(m)  # [1,H,W], float 0..1
                targets.append(m_t)

            targets = torch.stack(targets, dim=0).to(device)

            # compute metrics
            dice = dice_coeff(probs, targets).item()
            iou = iou_score(probs, targets, thr=threshold).item()
            total_dice += dice * b
            total_iou += iou * b
            n += b

    avg_dice = total_dice / max(1, n)
    avg_iou = total_iou / max(1, n)
    print(f"Evaluation: n={n}  Dice={avg_dice:.4f}  IoU={avg_iou:.4f}")
    return avg_dice, avg_iou


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run prediction using a trained UNet model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--out-masks-dir', type=str, required=True, help='Directory to save output masks.')
    parser.add_argument('--device', type=str, default=None, help='Device to run the prediction on (e.g., "cpu" or "cuda").')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for prediction.')
    parser.add_argument('--target-size', type=int, nargs=2, default=None, help='Optional target size H W to resize input images (e.g., --target-size 256 256)')

    args = parser.parse_args()

    run_prediction(
        model_path=args.model_path,
        images_dir=args.images_dir,
        out_masks_dir=args.out_masks_dir,
        device=args.device,
        batch_size=args.batch_size,
        target_size=tuple(args.target_size) if args.target_size else None
    )