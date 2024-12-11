import os
from datetime import datetime
from PIL import Image
from einops import rearrange 
from config import device
import torch
import numpy as np



@torch.no_grad()
def inpaint(diffusion, model, images, mask_known, labels=None):
    """Generate samples conditioned on known parts of images.
    
    Args:
      diffusion (Diffusion): The descriptor of a diffusion model.
      model: A denoising model: model(x, t, labels) outputs a denoised version of input x.
      images of shape (batch_size, n_channels, H, W): Conditioning images.
      mask_known of shape (batch_size, 1, H, W): BoolTensor which specifies known pixels in images (marked as True).
      labels of shape (batch_size,): Classes of images, None for no conditioning on classes.
    
    Returns:
      x of shape (batch_size, n_channels, H, W): Generated samples (one sample per input image).
    """
    # YOUR CODE HERE
    
    
    x_shape = images.shape
    x_t = torch.randn(x_shape).to(device)     
    

    for t in range(diffusion.num_timesteps,0,-1):

        # define epsilon here
        if t>1 :
            epsilon=torch.randn(x_shape).to(device) 
        else :
            epsilon=0
        # Update known pixels with noise
        alpha=torch.prod(1-diffusion.betas[:t])*torch.ones(x_shape[1:])
        x_t_minus1_known = torch.sqrt(alpha)*images+(1-alpha)*epsilon

        # define z here
        if t>1 :
            z=torch.randn(x_shape).to(device)  
        else :
            z=0    
        
        Ti=torch.ones(x_shape[0],device=device)*t
        #update unkown pixels with
        x_t_minus1_unknown = (1/(torch.sqrt(1-diffusion.betas[t-1]))) * (x_t - (diffusion.betas[t-1])*model(x_t,Ti) / torch.sqrt(1-alpha))+torch.sqrt(diffusion.betas[t-1])*z
        
        x_t = np.dot(x_t_minus1_known , mask_known) + np.dot(x_t_minus1_unknown, (~mask_known))

    # Returnx_0
    return x_t    
    
    
    
    
    
    
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
