
import numpy as np
import matplotlib.pyplot as plt
import json
def load_loss_history(loss_file):
    """Load training and validation loss history from a numpy file.

    Args:
        loss_file (str): Path to the .npz file containing 'train_losses' and 'val_losses'
    Returns:
        tuple: (train_losses, val_losses) as lists of floats
    """

    data = json.load(open(loss_file))
    val_losses = np.array(data['val_loss']).tolist()    
    train_losses = np.array(data['loss']).tolist()  

    return train_losses, val_losses  

def plot_loss(train_losses, val_losses, out_path='loss_plot.png'):
    """Plot training and validation loss curves and save to a file.

    Args:
        train_losses (list of float): List of training loss values per epoch.
        val_losses (list of float): List of validation loss values per epoch.
        out_path (str): Path to save the loss plot image.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Plot training and validation loss curves.")
    parser.add_argument('--loss-file', type=str, required=True, help='Path to the loss history .json file.')
    parser.add_argument('--out-path', type=str, default='loss_plot.png', help='Output path for the loss plot image.')
    args = parser.parse_args()

    train_losses, val_losses = load_loss_history(args.loss_file)
    plot_loss(train_losses, val_losses, args.out_path)
    print(f"Loss plot saved to {args.out_path}")
