import torch
import numpy as np

################################################################################################
# This module specifies the different types of noise schedules the diffusion model can use.
# Schedulers determine how noise is added and removed from the data at each step of the forward
#  (noising) and backward (denoising) processes.
################################################################################################



def linear_schedule(beta_min=0.0001, beta_max=0.02, T=1000):
    """
    Linear noise schedule.
    
    Args:
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.
        T (int): Total timesteps.
    
    Returns:
        torch.Tensor: A tensor of betas for each timestep.
    """
    return torch.linspace(beta_min, beta_max, T)





def cosine_schedule(T=1000, beta_min=0.0001, beta_max=0.02, s=0.008):
    """
    Cosine noise schedule with specified minimum and maximum values.

    Args:
        T (int): Total timesteps.
        s (float): Small constant for stability.
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.

    Returns:
        torch.Tensor: A tensor of betas for each timestep.
    """
    def alpha_bar(t):
        return np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2

    betas = []
    for t in range(T):
        if t == 0:
            betas.append(1 - alpha_bar(t))
        else:
            betas.append(min(1 - alpha_bar(t) / alpha_bar(t - 1), 0.999))  
    
    # Normalize betas to the range [0, 1]
    betas = np.array(betas)
    betas_normalized = (betas - betas.min()) / (betas.max() - betas.min())

    # Scale betas to the range [beta_min, beta_max]
    betas_scaled = beta_min + betas_normalized * (beta_max - beta_min)
    
    return torch.tensor(betas_scaled, dtype=torch.float32)






def quadratic_schedule(beta_min=0.0001, beta_max=0.02, T=1000):
    """
    Quadratic noise schedule.
    
    Args:
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.
        T (int): Total timesteps.
    
    Returns:
        torch.Tensor: A tensor of betas for each timestep.
    """
    timesteps = torch.linspace(0, 1, T)
    return beta_min + (beta_max - beta_min) * timesteps ** 2






def exponential_schedule(sigma_min=0.0001, sigma_max=0.02, T=1000):
    """
    Exponential noise schedule.

    Args:
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
        T (int): Total timesteps.

    Returns:
        np.ndarray: Exponential noise levels for each timestep.
    """
    t = torch.linspace(0, 1, T)  # Timesteps in range [0, 1]
    schedule = sigma_min * (sigma_max / sigma_min) ** t
    return schedule





def logarithmic_schedule(beta_min=0.0001, beta_max=0.02, T=1000, c=10):
    """
    Logarithmic noise schedule.

    Args:
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.
        T (int): Total timesteps.
        c (float): Scaling constant for the logarithm.

    Returns:
        np.ndarray: Logarithmic noise levels for each timestep.
    """
    timesteps = torch.linspace(0, T, T) 
    denominator = torch.log(torch.tensor(1 + c * T, dtype=torch.float32))
    normalized_log = torch.log(1 + c * timesteps) / denominator
    return beta_min + (beta_max - beta_min) * normalized_log


def select_betas(scheduler, beta_min, beta_max, T):

    betas = {
            'linear': linear_schedule(beta_min, beta_max, T),
            'cosine': cosine_schedule(T, beta_min, beta_max, s=0.008),
            'quadratic': quadratic_schedule(beta_min, beta_max, T),
            'exponential': exponential_schedule(beta_min, beta_max, T),
            'logarithmic': logarithmic_schedule(beta_min, beta_max, T, c=10),
        }.get(scheduler)
    
    return betas