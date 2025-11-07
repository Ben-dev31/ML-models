# unet_trainable.py

from pathlib import Path
from typing import Optional, Tuple, Any
import json

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import cv2

# ----------------------------
# Architecture U-Net
# ----------------------------
class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch: Optional[int] = None):
        """
        mid_ch: intermediate channels; if None, set to out_ch

        """
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
    def __init__(self, in_channels=3, out_channels=1, device=None, input_size: Tuple[int,int]=(256,256)):
        super(UNet, self).__init__()

        self.history = {}
        # store dynamic attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        
        # Encodeur
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Décodeur
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        # Sortie
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Gestion du GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Encodeur
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))

        # Décodeur
        x = self.upconv4(x5)
        x = self.conv_up4(torch.cat([x, x4], dim=1))
        x = self.upconv3(x)
        x = self.conv_up3(torch.cat([x, x3], dim=1))
        x = self.upconv2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        x = self.upconv1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))

        return self.final_conv(x)
    
    def save_history(self, history: dict, save_path: str):
        """
        Sauvegarde l'historique d'entraînement dans un fichier .pth

        Args:
            history (dict) : dictionnaire contenant l'historique (ex: {'loss': [...], 'val_loss': [...]})
            save_path (str) : chemin du fichier de sauvegarde
        """
        with open(save_path, "w") as f:
            json.dump(history, f)

    def train_model(self, train_loader,val_loader, epochs=10, lr=1e-4, criterion=None, 
                    save_dir: str = "./checkpoints", save_history_path: str = "training_history.json"):
        """
        Entraîne le modèle sur un DataLoader contenant des couples (image, masque).

        Args:
            train_loader : torch.utils.data.DataLoader
            epochs (int) : nombre d'époques
            lr (float) : taux d'apprentissage
            criterion : fonction de perte (par défaut BCEWithLogitsLoss)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = criterion or nn.BCEWithLogitsLoss()

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss_avg = epoch_loss / max(1, n_batches)
            val_loss = self.validate(val_loader, criterion)
            # append averaged epoch metrics
            self.history.setdefault('loss', []).append(epoch_loss_avg)
            self.history.setdefault('val_loss', []).append(val_loss)

            # checkpoint dict (compatible with load_checkpoint below)
           
            ckpt_path = save_dir / f"unet_epoch{epoch+1}.pth"
            torch.save(self.state_dict(), ckpt_path)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss_avg:.4f} - Val Loss: {val_loss:.4f}")

        print("Entraînement terminé.")
        # save history as JSON for human-readability
        self.save_history(self.history, save_history_path)
    
    def validate(self, val_loader, criterion=None):
        """
        Évalue le modèle sur un DataLoader de validation.

        Args:
            val_loader : torch.utils.data.DataLoader
            criterion : fonction de perte (par défaut BCEWithLogitsLoss)

        Returns:
            float : perte moyenne sur le jeu de validation
        """
        self.eval()
        criterion = criterion or nn.BCEWithLogitsLoss()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                outputs = self(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                n += 1
        return val_loss / max(1, n)
    
    def prepare_image(self, image_input, color_order: str = 'RGB'):
        """
        Prépare une image pour la prédiction.
        Accepte :
            - chemin vers une image (str)
            - numpy.ndarray (RGB ou BGR)
            - PIL.Image
        
        Retourne :
            torch.Tensor de forme [1, 3, H, W]
        """
        if isinstance(image_input, str):
            # Chargement depuis un chemin
            img = Image.open(image_input).convert("RGB")
            transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0)

        elif isinstance(image_input, np.ndarray):
            img = image_input
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            # Assume provided numpy is RGB by default. If BGR, user can set color_order='BGR'.
            if color_order.upper() == 'BGR' and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, tuple(self.input_size[::-1]))  # cv2 expects (w,h)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

        elif isinstance(image_input, Image.Image):
            transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor()
            ])
            img_tensor = transform(image_input).unsqueeze(0)

        else:
            raise TypeError("Format d'image non reconnu. Utilise un chemin, un numpy.ndarray ou un PIL.Image.")

        return img_tensor

    
    def predict(self, image_input, color_order: str = 'RGB') -> Any:
        """
        Prédit un masque à partir d'une image.
        Accepte :
            - chemin vers une image
            - numpy.ndarray
            - PIL.Image
        """
        self.eval()
        with torch.no_grad():
            # Préparer l'image automatiquement
            img_tensor = self.prepare_image(image_input, color_order=color_order).to(self.device)

            output = self(img_tensor)
            probs = torch.sigmoid(output)
           
            return probs

