
from unet import UNet, SegmentationDataset
from losses_metrics import DiceLoss, dice_coeff, iou_score
import torch

from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import PredictionDataset, save_in_history
from typing import Optional, Tuple
from model_test import run_prediction, evaluate_model
from model_full_train import run_full_train
from losses_metrics import FoclssLoss, DiceLoss
from plot_loss import plot_loss , load_loss_history

# pipeline to evaluate model on a dataset and compute metrics
# load dataset -> trainning model -> evaluate metrics -> test prediction saving -> save metrics to history


def evaluate_model(data_path: str,
                   epochs: int = 5,
                   batch_size: int = 2,
                   device: Optional[str] = None,
                   loss_fns: Optional[dict] = {'dice': DiceLoss()}
                   ):
    

    # load dataset
    train_dataset_path = Path(data_path) / 'train'
    val_dataset_path = Path(data_path) / 'val'
    test_dataset_path = Path(data_path) / 'test'

    for loss_name, loss_fn in loss_fns.items():
        print(f"Using loss function for training: {loss_name} -> {loss_fn}")
        # train model
        run_full_train(root=data_path, 
                    train_dir=str(train_dataset_path), 
                    val_dir=str(val_dataset_path), 
                    epochs=epochs,
                    batch_size=batch_size,
                    save_dir=str(Path(data_path).parent / f'model_checkpoints-{loss_name}'),
                    device=device,
                    criterion= loss_fn,
                    history_path=str(Path(data_path).parent / f'training_history_{loss_name}.json')
            )
        
        # plot training history
        loss_file = str(Path(data_path).parent / f'training_history_{loss_name}.json')
        train_losses, val_losses = load_loss_history(loss_file)
        plot_loss(train_losses, val_losses, 
                  out_path=str(Path(data_path).parent / f'loss_plot_{loss_name}.png'),
                  metric_name=loss_name)

        # test prediction
        model_ckpt = str(Path(data_path).parent.joinpath(f'model_checkpoints-{loss_name}/unet_epoch{epochs -1}.pth'))
        out_masks_dir = str(Path(data_path) / 'test' / f'predicted_masks-{loss_name}')
        run_prediction(model_path=model_ckpt,
                    images_dir=str(test_dataset_path / 'images'),
                        out_masks_dir=out_masks_dir,
                        device=device,
                        batch_size=batch_size,
                        target_size=(256, 256),
                        metrics={'dice': dice_coeff, 'iou': iou_score}
                        )
        
    print("Evaluation completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on dataset with different loss functions.")
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training (e.g., "cuda" or "cpu").')     
    args = parser.parse_args()
    evaluate_model(data_path=args.data_path,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   device=args.device,
                     loss_fns={
                          'dice': DiceLoss(),
                          'focls': FoclssLoss()
                     }  
                   )


   