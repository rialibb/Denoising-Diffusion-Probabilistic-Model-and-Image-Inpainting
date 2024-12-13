import torch
from dataset import DiffSet
from schedules.schedules import linear_schedule, select_betas
from torch.utils.data import DataLoader, random_split
from diffusion_model import Diffusion, UNet, InPaint
from train_test import f_train, f_test
from config import device
from tools import save_images, plot_losses, SaveBestModelCallback
import optuna
import os
import joblib




def load_data(dataset_choice, batch_size):

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




def run_training_and_testing_pipeline(
    T = 1000,
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 20,
    batch_size = 128,
    lr = 0.001,
    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    beta_min = 0.0001,
    beta_max = 0.02,
    s = 0.008,
    c = 10,
    save_dir: str = 'saved_models'
    ):


    # Load data
    train_loader, val_loader, test_loader, test_dataset = load_data(dataset_choice, batch_size)

    # Lookup table to set scheduler
    betas = select_betas(scheduler, beta_min, beta_max, T)

    # Create models
    diffusion = Diffusion(betas, T)
    unet = UNet(
        img_channels = test_dataset.depth,
        base_channels = test_dataset.size,
        time_emb_dim = test_dataset.size,
        num_classes = None,
    )
    diffusion.to(device)
    unet.to(device)

    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # create the save path for the model
    save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_diffusion.pth')
    save_path_unet = os.path.join(save_dir, f'{dataset_choice}_unet.pth')

    # Train models
    train_losses, val_losses = f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)
    # Test models
    test_loss = f_test(diffusion, unet, test_loader)
    # Save best models
    torch.save(diffusion.state_dict(), save_path_diffusion)
    torch.save(unet.state_dict(), save_path_unet)

    plot_losses(train_losses, val_losses)




def run_hyperparam_tuning_pipeline(
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 15,
    batch_size = 128,
    num_trials = 5,
    T= 1000,
    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    ):

    # Load datasets
    train_loader, val_loader, test_loader, test_dataset = load_data(dataset_choice, batch_size)

    # Objective function for Optuna
    def objective(trial):

        # Hyperparameters to tune
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        beta_min = trial.suggest_float('beta_min', 1e-5, 0.01)
        beta_max = trial.suggest_float('beta_max', 0.01, 0.05)
        # Set up diffusion process
        # Lookup table to set scheduler
        betas = select_betas(scheduler, beta_min, beta_max, T)
        
        # Create diffusion and UNet models
        diffusion = Diffusion(betas, num_timesteps=T)
        unet = UNet(
            img_channels=test_dataset.depth,
            base_channels=test_dataset.size,
            time_emb_dim=test_dataset.size,
            num_classes=None,
        )
        
        diffusion.to(device)
        unet.to(device)

        # Train the model
        train_losses, val_losses = f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)
        
        # Test the model
        test_loss = f_test(diffusion, unet, test_loader)

        # Save only best model
        save_best_model = SaveBestModelCallback()
        save_best_model(diffusion, unet, val_losses[-1], dataset_choice)
    
        return val_losses[-1]
    

    # Set up the Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    print("Best trial:")
    print(f"  Value (Validation Loss): {study.best_trial.value}")
    print(f"  Hyperparameters: {study.best_trial.params}")

    # Save the study results in a file for later use
    joblib.dump(study, 'optuna_study.pkl')




def run_inpainting_pipeline(
    T = 1000,
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    batch_size = 128,
    beta_min = 0.0001,
    beta_max = 0.02,
):

    _, _, _, test_dataset = load_data(dataset_choice, batch_size)

    betas = linear_schedule(beta_min, beta_max, T)

    # Create models
    diffusion = Diffusion(betas, T)
    unet = UNet(
        img_channels = test_dataset.depth,
        base_channels = test_dataset.size,
        time_emb_dim = test_dataset.size,
        num_classes = None,
    )
    diffusion.to(device)
    unet.to(device)

    diffusion.load_state_dict(torch.load(f'saved_models/{dataset_choice}_best_diffusion.pth'))
    unet.load_state_dict(torch.load(f'saved_models/{dataset_choice}_best_unet.pth'))
    
        
    # Sample generation
    x_shape = (25, test_dataset.depth, test_dataset.size, test_dataset.size)
    samples = diffusion.sample(unet, x_shape)
    samples01 = ((samples + 1) / 2).clip(0, 1)
    save_images(samples01, dataset_choice, save_dir='generated_samples', image_type = 'samples' , cmap='binary', ncol=5)


    # Inpainting masked image
    image = test_dataset[5643]  # Select one image from the test dataset

    inpaint = InPaint()
    inpaint.sample(diffusion, unet, image, dataset_choice)







































    
