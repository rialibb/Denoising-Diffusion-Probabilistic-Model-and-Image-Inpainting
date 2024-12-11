
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import datetime
from models import Diffusion, UNet
import logging
from torch.utils.data import DataLoader
from config import device
import numpy as np
import tools
from pytorch_fid import fid_score




def notify(*msg: str):
    """ print logs in terminal

    Parameters
    ----------
    msg: str
        the message to print
    """
    message = " ".join(msg)
    print(message)

notify("Job started!")


def f_train(
    diffusion: nn.Module,
    unet: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    n_epochs: int = 20,
    learning_rate: float = 0.001,
    logging_file: str = "training.log",
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
    logging_file: str
        path to the file to load logs 
    """

    # retrieve current Date/time
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notify(f"{date}")
    logging.basicConfig(filename=logging_file)

    # We use cross-entropy as it is well-known for performing well in classification problems
    diffusion.to(device)
    unet.to(device)
    loss_func = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(unet.parameters(),lr=learning_rate)
    scheduler = OneCycleLR(optimizer,max_lr=learning_rate, epochs = n_epochs, steps_per_epoch= len(trainloader), pct_start = 0.1)

    # initialize parameters
    early_stopping_patience = 4
    previous_val_loss = float('inf')
    count_no_improvement = 0

    notify(f"----------------------- Starting training -----------------------")
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
        
        real_images = []
        generated_images = []

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
                
                # add batch images 
                real_images.append(images.cpu())
                # add generated images
                generated_samples = diffusion.sample(unet, images.shape).cpu()
                generated_images.append(generated_samples)
                
        avg_val_loss = np.mean(val_loss)
        
        # Flatten lists into tensors
        real_images = torch.cat(real_images)
        generated_images = torch.cat(generated_images)

        # Normalize images to [0, 1] for FID calculation
        real_images = ((real_images + 1) / 2).clamp(0, 1)
        generated_images = ((generated_images + 1) / 2).clamp(0, 1)
        #calculate bthe FID score
        FID = fid_score.calculate_from_tensors(real_images, generated_images, device=device)

        notify(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation FID: {FID:.4f},  Last value of learning rate for this epoch: {scheduler.get_last_lr()}")

        # Check for improvement
        if avg_val_loss < previous_val_loss:
            previous_val_loss = avg_val_loss
            count_no_improvement = 0
            # Save the best model
            torch.save(unet.state_dict(), "best_unet_model.pth")
            print(f"Validation loss improved. Model saved.")
        else:
            count_no_improvement += 1
            print(f"No improvement in validation loss for {count_no_improvement} epoch(s).")
        
        # Early stopping
        if count_no_improvement >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement in validation loss for {early_stopping_patience} epochs.")
            break

    # save the model
    tools.save_model(diffusion, 'saved_models/diffusion.pth', confirm=True)
    tools.save_model(unet, 'saved_models/unet.pth', confirm=True)
    notify("Model saved")

    notify("----------------------FINISHED TRAINING----------------------")
    
    
    

    

def f_test(
    diffusion: nn.Module,
    unet: nn.Module,
    testloader: DataLoader,
    logging_file: str = "test.log",
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
    logging_file: str
        path to the file to load logs 
    """
    loss_func = torch.nn.MSELoss()
    diffusion.eval()
    unet.eval()
    test_loss = []
    
    real_images = []
    generated_images = []

    notify(f"----------------------- Starting TESTING -----------------------")
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
            
            # add batch images 
            real_images.append(images.cpu())
            # add generated images
            generated_samples = diffusion.sample(unet, images.shape).cpu()
            generated_images.append(generated_samples)
            
    avg_test_loss = np.mean(test_loss)
    
    # Flatten lists into tensors
    real_images = torch.cat(real_images)
    generated_images = torch.cat(generated_images)

    # Normalize images to [0, 1] for FID calculation
    real_images = ((real_images + 1) / 2).clamp(0, 1)
    generated_images = ((generated_images + 1) / 2).clamp(0, 1)
    #calculate bthe FID score
    FID = fid_score.calculate_from_tensors(real_images, generated_images, device=device)

    notify(f"Test Loss: {avg_test_loss:.4f}, Test FID: {FID:.4f}")
    notify("----------------------FINISHED TESTING----------------------")
