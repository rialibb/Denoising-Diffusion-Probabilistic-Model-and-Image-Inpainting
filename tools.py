import os
from PIL import Image
from einops import rearrange 
import torch
import matplotlib.pyplot as plt



def save_images(images, dataset_choice, save_dir='generated_samples', image_type = 'samples', ncol=5, **kwargs):
    """
    Save generated images to a folder instead of displaying them.

    Args:
        images (torch.Tensor): Generated images in tensor format.
        save_dir (str): Directory to save the images.
        image_type(str): whether the original image, the masked image or the generated image
        ncol (int): Number of columns for arranging images in a grid.
    """
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename using the current timestamp
    save_path = os.path.join(save_dir, f'{dataset_choice}_{image_type}.png')

    # Rearrange the images into a grid
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    if out.shape[0] == 1:
        image = out[0].numpy()  # For grayscale images
        Image.fromarray((image * 255).astype('uint8')).save(save_path)
    else:
        image = out.permute((1, 2, 0)).numpy()  # For RGB images
        Image.fromarray((image * 255).astype('uint8')).save(save_path)

    print(f"Image saved to {save_path}")



def plot_losses(val_losses, train_losses):
    """
    Plots training and validation loss over epochs.

    Args:
        val_losses (list): Validation loss values for each epoch.
        train_losses (list): Training loss values for each epoch.
    """

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



class SaveBestModelCallback:
    """Custom callback for saving the best model during Optuna optimization."""
    def __init__(self):
        self.best_val_loss = float('inf')  # Start with a very large validation loss

    def __call__(self, diffusion, unet, val_loss, dataset_choice):
        """Check if the current validation loss is the best, and if so, save the model."""
        if val_loss < self.best_val_loss:
            save_dir = 'saved_models'
            os.makedirs(save_dir, exist_ok=True)
            save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_best_diffusion.pth')
            save_path_unet = os.path.join(save_dir, f'{dataset_choice}_best_unet.pth')

            print(f"New best model found! Saving model with val_loss: {val_loss:.4f}")
            self.best_val_loss = val_loss
            torch.save(diffusion.state_dict(), save_path_diffusion)
            torch.save(unet.state_dict(), save_path_unet)



