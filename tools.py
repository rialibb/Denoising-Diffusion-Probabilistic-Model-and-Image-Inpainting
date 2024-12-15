import os
from PIL import Image
from einops import rearrange 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import DiffSet







def load_data(dataset_choice, batch_size):
    """
    Loads and splits the dataset into train, validation, and test sets, and returns DataLoaders.

    Args:
        dataset_choice (str): The name of the dataset to load (e.g., 'CelebA' or other available datasets).
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: DataLoaders for train, validation, and test sets, along with the test dataset.
    """

    if dataset_choice == "CelebA":
        train_dataset = DiffSet('train', dataset_choice)
        val_dataset = DiffSet('valid', dataset_choice)
        test_dataset = DiffSet('test', dataset_choice)
        
    else:
        train_val_dataset = DiffSet(True, dataset_choice)
        test_dataset = DiffSet(False, dataset_choice)

        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, test_dataset








def save_images(images, dataset_choice, save_dir='generated_samples', image_type='samples', ncol=5, scale_factor=2, **kwargs):
    """
    Save generated images to a folder with increased resolution.

    Args:
        images (torch.Tensor): Generated images in tensor format.
        save_dir (str): Directory to save the images.
        dataset_choice (str): Name of the dataset being processed.
        image_type (str): Type of image ('samples', 'masked', etc.).
        ncol (int): Number of columns for arranging images in a grid.
        scale_factor (int): Factor by which to upscale the image resolution.
    """
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename using the current timestamp
    save_path = os.path.join(save_dir, f'{dataset_choice}_{image_type}.png')

    # Rearrange the images into a grid
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    
    # Convert the tensor to a NumPy array and upscale
    if out.shape[0] == 1:  # For grayscale images
        image = out[0].numpy()  # Extract grayscale channel
        image = Image.fromarray((image * 255).astype('uint8'))
    else:  # For RGB images
        image = out.permute((1, 2, 0)).numpy()
        image = Image.fromarray((image * 255).astype('uint8'))
    
    # Resize image to increase resolution
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)  # Scale dimensions
    image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    
    # Save the upscaled image
    image.save(save_path)
    print(f"Image saved to {save_path} with resolution {new_size}")








def plot_losses(all_train_losses, all_val_losses, schedulers, save_dir='plots'):
    """
    Saves two plots: 
    1. Training loss evolution for different configurations.
    2. Validation loss evolution for different configurations.

    Args:
        all_train_losses (list of lists): Training losses for multiple configurations.
        all_val_losses (list of lists): Validation losses for multiple configurations.
        save_dir (str): Directory where plots will be saved.
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get the number of epochs (assuming all configurations have same epoch count)
    epochs = list(range(1, len(all_train_losses[0]) + 1))

    # Plot training loss evolution for different configurations
    plt.figure(figsize=(10, 6))
    for scheduler, train_losses in zip(schedulers, all_train_losses):
        plt.plot(epochs, train_losses, label=f'{scheduler} scheduler', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Evolution for Different Schedulers')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the training loss plot
    train_loss_path = os.path.join(save_dir, 'training_loss_evolution.png')
    plt.savefig(train_loss_path)
    plt.close()  # Close the figure to avoid overlap in subsequent plots
    print(f"Training loss plot saved to {train_loss_path}")

    # Plot validation loss evolution for different configurations
    plt.figure(figsize=(10, 6))
    for scheduler, val_losses in zip(schedulers, all_val_losses):
        plt.plot(epochs, val_losses, label=f'{scheduler} scheduler', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Evolution for Different Configurations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the validation loss plot
    val_loss_path = os.path.join(save_dir, 'validation_loss_evolution.png')
    plt.savefig(val_loss_path)
    plt.close()
    print(f"Validation loss plot saved to {val_loss_path}")









class SaveBestModelCallback:
    """Custom callback for saving the best model during Optuna optimization."""
    def __init__(self):
        self.best_val_loss = float('inf')  # Start with a very large validation loss

    def __call__(self, diffusion, unet, val_loss, dataset_choice, scheduler):
        """Check if the current validation loss is the best, and if so, save the model."""
        if val_loss < self.best_val_loss:
            save_dir = 'saved_models'
            os.makedirs(save_dir, exist_ok=True)
            save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_{scheduler}_best_diffusion.pth')
            save_path_unet = os.path.join(save_dir, f'{dataset_choice}_{scheduler}_best_unet.pth')

            print(f"New best model found! Saving model with val_loss: {val_loss:.4f}")
            self.best_val_loss = val_loss
            torch.save(diffusion.state_dict(), save_path_diffusion)
            torch.save(unet.state_dict(), save_path_unet)



