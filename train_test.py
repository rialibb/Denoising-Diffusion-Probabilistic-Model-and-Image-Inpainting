
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import datetime
from models import Diffusion, UNet
import logging
from torch.utils.data import DataLoader
from config import device
import numpy as np
from pytorch_fid import fid_score
import sys
import os






print("Job started!")


def f_train(
    diffusion: nn.Module,
    unet: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    n_epochs: int = 20,
    learning_rate: float = 0.001,
    dataset_choice: str='MNIST',
    save_dir: str = 'saved_models'
):
    """Train the diffusion model to predict the noise

    Parameters
    ----------
    diffusion: torch.nn.Module
        the diffusion model
    unet: torch.nn.Module
        model to predixt noise for diffusion 
    trainset: DataLoader
        the training dataloader 
    valloader: DataLoader
        the validation dataloader 
    n_epochs: int
        number of epchs for training
    learning_rate: float
        the learning rate to use for training
    dataset_choice: str
        the dataset to consider to train the diffusion model
    save_dir: str
        the directory to save the model
    """
    logging.basicConfig(
        stream=sys.stdout,  
        level=logging.INFO,
        format="%(message)s"
    )
    
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # create the save path for the model
    save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_diffusion.pth')
    save_path_unet = os.path.join(save_dir, f'{dataset_choice}_unet.pth')
    
    # retrieve current Date/time
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{date}")

    # We use cross-entropy as it is well-known for performing well in classification problems
    loss_func = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(unet.parameters(),lr=learning_rate)
    scheduler = OneCycleLR(optimizer,max_lr=learning_rate, epochs = n_epochs, steps_per_epoch= len(trainloader), pct_start = 0.1)

    # initialize parameters
    early_stopping_patience = 4
    previous_val_loss = float('inf')
    count_no_improvement = 0

    logging.info(f"----------------------- Starting training -----------------------")
    for epoch in range(n_epochs):
        
        # training Phase
        diffusion.train()
        unet.train()
        train_loss = []
        
        for images in trainloader :
            images=images.to(device)
            t=torch.linspace(0, diffusion.num_timesteps-1, images.shape[0]).to(device)   
            epsilon=torch.randn(images.shape).to(device)
            
            optimizer.zero_grad()
            
            x_t=diffusion.forward(images,t,epsilon)
            noise=unet.forward(x_t,time=t)

            loss = loss_func(noise,epsilon)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
        avg_train_loss = np.mean(train_loss)
        

        # Validation Phase
        diffusion.eval()
        unet.eval()
        val_loss = []

        with torch.no_grad():
            for images in valloader:  
                images = images.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
                epsilon = torch.randn(images.shape, device=device)
                
                # Forward pass
                x_t = diffusion.forward(images, t, epsilon)
                predicted_noise = unet.forward(x_t, time=t)
                
                # Compute loss
                loss = loss_func(predicted_noise, epsilon)
                val_loss.append(loss.item())
        avg_val_loss = np.mean(val_loss)

        logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Last value of learning rate for this epoch: {scheduler.get_last_lr()}")

        # Check for improvement
        if avg_val_loss < previous_val_loss:
            previous_val_loss = avg_val_loss
            count_no_improvement = 0
            # Save the best model
            torch.save(diffusion.state_dict(), save_path_diffusion)
            torch.save(unet.state_dict(), save_path_unet)
            print(f"Validation loss improved. Models saved")
        else:
            count_no_improvement += 1
            print(f"No improvement in validation loss for {count_no_improvement} epoch(s).")
        
        # Early stopping
        if count_no_improvement >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement in validation loss for {early_stopping_patience} epochs.")
            break

    logging.info("----------------------FINISHED TRAINING----------------------")
    
    
    

    

def f_test(
    diffusion: nn.Module,
    unet: nn.Module,
    testloader: DataLoader
):
    """test the diffusion model on the test loader

    Parameters
    ----------
    diffusion: torch.nn.Module
        the diffusion model
    unet: torch.nn.Module
        model to predixt noise for diffusion 
    testloader: DataLoader
        the test dataloader 
    """
    
    logging.basicConfig(
        stream=sys.stdout,  
        level=logging.INFO,
        format="%(message)s"
    )
        
    loss_func = torch.nn.MSELoss()
    diffusion.eval()
    unet.eval()
    test_loss = []

    logging.info(f"----------------------- Starting TESTING -----------------------")
    with torch.no_grad():
        for images in testloader:  
            images = images.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            epsilon = torch.randn(images.shape, device=device)
            
            # Forward pass
            x_t = diffusion.forward(images, t, epsilon)
            predicted_noise = unet.forward(x_t, time=t)
            
            # Compute loss
            loss = loss_func(predicted_noise, epsilon)
            test_loss.append(loss.item())
    avg_test_loss = np.mean(test_loss)

    logging.info(f"Test Loss: {avg_test_loss:.4f}")
    logging.info("---------------------- FINISHED TESTING----------------------")
