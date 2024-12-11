import torch
from data import DiffSet
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
import imageio
import glob
from models import Diffusion, UNet
from train_test import f_train, f_test
from config import device


# skip training
skip_training = False


# Training hyperparameters
num_timesteps = 1000
dataset_choice = "MNIST"
beta_min = 0.0001
beta_max = 0.02
n_epochs = 20
batch_size = 128
lr = 0.001

# load dataset
train_val_dataset = DiffSet(True, dataset_choice)
test_dataset = DiffSet(False, dataset_choice)

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)





# Create models
diffusion = Diffusion(1000)
unet = UNet(
    img_channels=1,
    base_channels=32,
    time_emb_dim=32,
    num_classes=None,
)
diffusion.to(device)
unet.to(device)




# train the model
if not skip_training:
    # training the models
    f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)
    # testing the models
    f_test(diffusion, unet, test_loader)

else:
    diffusion.load_state_dict(torch.load('saved_models/diffusion.pth'))
    unet.load_state_dict(torch.load('saved_models/unet.pth'))
    