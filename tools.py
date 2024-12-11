import os
from datetime import datetime
from PIL import Image
from einops import rearrange  # Ensure you have this library installed: `pip install einops`

def save_images(images, dataset_choice, save_dir='generated_samples', ncol=10, **kwargs):
    """
    Save generated images to a folder instead of displaying them.

    Args:
        images (torch.Tensor): Generated images in tensor format.
        save_dir (str): Directory to save the images.
        ncol (int): Number of columns for arranging images in a grid.
    """
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{dataset_choice}_samples_{timestamp}.png')

    # Rearrange the images into a grid
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    if out.shape[0] == 1:
        image = out[0].numpy()  # For grayscale images
        Image.fromarray((image * 255).astype('uint8')).save(save_path)
    else:
        image = out.permute((1, 2, 0)).numpy()  # For RGB images
        Image.fromarray((image * 255).astype('uint8')).save(save_path)

    print(f"Image saved to {save_path}")
