
import os
from typing import Optional, Tuple
from unet import UNet, SegmentationDataset
from losses_metrics import DiceLoss
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