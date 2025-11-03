# unet_trainable.py
import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# 1) Architecture U-Net
# ----------------------------
class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch: Optional[int] = None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv. Use ConvTranspose2d for learnable upsample."""
    def __init__(self, in_ch, out_ch, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        # x1: decoder feature, x2: skip connection from encoder
        x1 = self.up(x1)
        # pad if needed (when image sizes are odd)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2,
                                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, base_c: int = 64, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.down4 = Down(base_c*8, base_c*8)  # bottom (no further increase)

        self.up1 = Up(base_c*16, base_c*4, bilinear)
        self.up2 = Up(base_c*8, base_c*2, bilinear)
        self.up3 = Up(base_c*4, base_c, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # [B, base_c, H, W]
        x2 = self.down1(x1)   # [B, base_c*2, H/2, W/2]
        x3 = self.down2(x2)   # ...
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits  # raw logits; apply sigmoid for binary

# ----------------------------
# 2) Dataset
# ----------------------------
class SegmentationDataset(Dataset):
    """
    Dataset that expects two folders: images/ and masks/
    - images: RGB images (png/jpg/tiff)
    - masks: single-channel masks (0 background, 255 object) or multi-class index maps
    Both must have same filenames.
    """
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.images_dir / fname
        mask_path = self.masks_dir / fname  # assumes same name; adapt if suffix differs
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask
        if self.transform:
            sample = self.transform(image=np.array(image), mask=np.array(mask))
            image = sample['image']
            mask = sample['mask']
        else:
            # minimal transform: to tensor + normalize
            tf = transforms.Compose([
                transforms.ToTensor(),  # [0,1]
            ])
            image = tf(image)
            mask = np.array(mask, dtype=np.uint8)
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # [0,1], shape [1,H,W]

        return image, mask

# Example of a simple torchvision transform (no albumentations):
def make_basic_transform(target_size: Tuple[int,int]=(256,256)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # image -> [C,H,W] float 0..1
    ])
# If you want synchronized augmentations for image+mask, prefer albumentations (not used here).

# ----------------------------
# 3) Losses & Metrics
# ----------------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    # pred and target are tensors with values in [0,1], shape [B,1,H,W]
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        # logits: raw (not sigmoid). target in {0,1}
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, target, eps=self.eps)
        return 1.0 - dice

def iou_score(pred: torch.Tensor, target: torch.Tensor, thr=0.5, eps=1e-7):
    pred_bin = (pred > thr).float()
    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    target_flat = target.view(B, -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()

# ----------------------------
# 4) Training loop
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, criterion_bce, criterion_dice, epoch, log_interval=50):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)  # shape [B,1,H,W]

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            # if logits shape [B, C, H, W], for binary keep C=1
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print(f"Epoch {epoch} Step {i+1}/{len(loader)} - loss: {running_loss / (i+1):.4f}")

    avg_loss = running_loss / len(loader)
    return avg_loss

def validate(model, loader, device, criterion_bce, criterion_dice):
    model.eval()
    val_loss = 0.0
    dice_meter = 0.0
    iou_meter = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            logits = model(images)
            loss = 0.5 * criterion_bce(logits, masks) + 0.5 * criterion_dice(logits, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            dice_meter += dice_coeff(probs, masks).item()
            iou_meter += iou_score(probs, masks).item()

    n = len(loader)
    return val_loss / n, dice_meter / n, iou_meter / n

# ----------------------------
# 5) Run training (example)
# ----------------------------
def run_training(
    images_train_dir: str,
    masks_train_dir: str,
    images_val_dir: str,
    masks_val_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: Optional[str] = None,
    save_dir: str = "./checkpoints",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(save_dir, exist_ok=True)

    # Datasets & loaders (you can replace transforms with albumentations for stronger augmentation)
    train_ds = SegmentationDataset(images_train_dir, masks_train_dir, transform=None)
    val_ds = SegmentationDataset(images_val_dir, masks_val_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=3, n_classes=1, base_c=64, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion_bce, criterion_dice, epoch)
        val_loss, val_dice, val_iou = validate(model, val_loader, device, criterion_bce, criterion_dice)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} -> train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_dice: {val_dice:.4f} | val_iou: {val_iou:.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'val_loss': val_loss
        }
        torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))
            print("Saved best checkpoint.")

    print("Training finished.")

# If you want to load a checkpoint:
def load_checkpoint(model: nn.Module, ckpt_path: str, optimizer: Optional[torch.optim.Optimizer]=None, scaler: Optional[GradScaler]=None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', None), ckpt.get('val_loss', None)
