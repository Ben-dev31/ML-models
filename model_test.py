
import os
from typing import Optional, Tuple
from unet import UNet
from losses_metrics import dice_coeff, iou_score
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import PredictionDataset, save_in_history



def run_prediction(model_path: str, images_dir: str, 
                   out_masks_dir: str, 
                   device=None, 
                   batch_size=2, 
                   target_size: Optional[Tuple[int,int]] = None,
                   metrics: Optional[dict] = None,
                   histrory_path: Optional[str] = "prediction_history.json"
                   ):
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
    # Evaluate if ground-truth masks are available
    gt_masks_dir = Path(images_dir).parent / 'masks'
    if gt_masks_dir.exists():
        print("Ground-truth masks directory found, running evaluation...")
        evaluate_model(model_path, images_dir, 
                       str(gt_masks_dir), 
                       device=device, 
                       batch_size=batch_size, 
                       target_size=model_target,
                       metrics=metrics,
                       history_path=histrory_path
                       )
    
    print(f"Prediction completed. Masks saved to {out_masks_dir}")


def evaluate_model(model_path: str, 
                   images_dir: str,
                    masks_dir: str, 
                    device=None, 
                   batch_size: int = 4, 
                   threshold: float = 0.5, 
                   target_size: Optional[Tuple[int,int]] = None,
                   metrics: Optional[dict] = None,
                   save_results: bool = True,
                   history_path: Optional[str] = "loss_history.json"
                   ):
    """Evaluate model predictions against ground-truth masks.

    Returns average Dice and IoU across dataset.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model on device: {device}")
    history = {}
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
            if metrics is None or 'dice' in metrics:
                dice = dice_coeff(probs, targets).item()
                iou = iou_score(probs, targets, thr=threshold).item()
                total_dice += dice * b
                total_iou += iou * b
                n += b
                history['dice'] = history.get('dice', 0.0) + dice * b
                history['iou'] = history.get('iou', 0.0) + iou * b

            else:
                for metric_name, metric_fn in metrics.items():
                    score = metric_fn(probs, targets, thr = threshold).item()
                    history[metric_name] = history.get(metric_name, 0.0) + score * b

    for key in history:
        history[key] = history[key] / max(1, n) 
    
    if save_results or history_path:
        save_in_history(history, history_path)
        print(f"Saved evaluation results to {history_path}")

    print(f"Evaluation completed over {n} samples. Results: {history}")


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