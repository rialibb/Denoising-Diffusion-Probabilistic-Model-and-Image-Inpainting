import torch
from dataset import DiffSet
import pytorch_lightning as pl
from schedules.schedules import linear_schedule, cosine_schedule, quadratic_schedule, exponential_schedule, logarithmic_schedule
from torch.utils.data import DataLoader, random_split
from diffusion_model import Diffusion, UNet
from train_test import f_train, f_test
from config import device
from tools import save_images


# skip training
skip_training = True



# Training hyperparameters

num_timesteps = 1000
dataset_choice = "MNIST"   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
beta_min = 0.0001
beta_max = 0.02
n_epochs = 20
batch_size = 128
lr = 0.001



# define the schedule type
betas = linear_schedule(beta_min=beta_min, beta_max=beta_max, T=num_timesteps)



# load dataset
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





# Create models
diffusion = Diffusion(betas, 1000)
unet = UNet(
    img_channels = test_dataset.depth,
    base_channels = test_dataset.size,
    time_emb_dim = test_dataset.size,
    num_classes = None,
)
diffusion.to(device)
unet.to(device)




# train the model
if not skip_training:
    # training the models
    f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr, dataset_choice=dataset_choice)
    # testing the models
    f_test(diffusion, unet, test_loader)

else:
    # import trained model
    diffusion.load_state_dict(torch.load(f'saved_models/{dataset_choice}_diffusion.pth'))
    unet.load_state_dict(torch.load(f'saved_models/{dataset_choice}_unet.pth'))
    
    
# Sample generation
x_shape = (100, test_dataset.depth, test_dataset.size, test_dataset.size)
samples = diffusion.sample(unet, x_shape)
samples01 = ((samples + 1) / 2).clip(0, 1)
save_images(samples01, dataset_choice, save_dir='generated_samples', cmap='binary', ncol=10)