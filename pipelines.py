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
    """
    Runs the training and testing pipeline for a chosen configuration of parameters.

    Args:
        T (int): Total number of time steps for the diffusion process.
        dataset_choice (str): Dataset to use.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for the DataLoaders.
        lr (float): Learning rate for the optimizer.
        scheduler (str): Beta schedule.
        beta_min (float): Minimum beta value for the scheduler.
        beta_max (float): Maximum beta value for the scheduler.
        s (float): Stability constant if cosine scheduler used.
        c (float): Scaling constant if logarithmic scheduler used.
        save_dir (str): Directory to save the trained models.

    The function performs the following steps:
    1. Loads the specified dataset and splits it into train, validation, and test sets.
    2. Initializes the scheduler, diffusion model, and UNet model.
    3. Trains the models using the training and validation sets.
    4. Tests the models using the test set.
    5. Saves the trained models to the specified directory.
    6. Plots training and validation loss across epochs.
    """

    # Load data
    train_loader, val_loader, test_loader, test_dataset = load_data(dataset_choice, batch_size)

    # select scheduler
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
    # Plot losses through epochs
    plot_losses(train_losses, val_losses)




def run_hyperparam_tuning_pipeline(
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 15,
    batch_size = 128,
    num_trials = 5,
    T= 1000,
    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    ):
    """
    Runs hyperparameter tuning using Bayesian Optimization, with Optuna.

    Args:
        dataset_choice (str): Dataset to use.
        n_epochs (int): Number of epochs for training in each trial.
        batch_size (int): Batch size for the DataLoaders.
        num_trials (int): Number of trials to run for hyperparameter tuning.
        T (int): Total number of time steps for the diffusion process.
        scheduler (str): Beta scheduler.

    The function performs the following steps:
    1. Loads the dataset and splits it into train, validation, and test sets.
    2. Uses Optuna to optimize hyperparameters for the learning rate, beta_min, and beta_max.
    3. Trains a diffusion model and a UNet for each trial.
    4. Tests the model and saves the best-performing models.
    5. Saves the Optuna study results to a file for future reference.
    """

    # Load datasets
    train_loader, val_loader, test_loader, test_dataset = load_data(dataset_choice, batch_size)

    # Objective function for Optuna
    def objective(trial):

        # Hyperparameters to tune
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        beta_min = trial.suggest_float('beta_min', 1e-5, 0.01)
        beta_max = trial.suggest_float('beta_max', 0.01, 0.05)
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
    """
    Runs the inpainting pipeline using a diffusion model and UNet.

    Args:
        T (int): Total number of time steps for the diffusion process.
        dataset_choice (str): Dataset to use.
        batch_size (int): Batch size for loading the dataset.
        beta_min (float): Minimum beta value for the linear scheduler.
        beta_max (float): Maximum beta value for the linear scheduler.

    The function performs the following steps:
    1. Loads the test dataset for the specified dataset choice.
    2. Initializes and loads pre-trained weights for the diffusion and UNet models.
    3. Generates and saves a batch of sample images from the diffusion model.
    4. Selects a single test image, performs image inpainting and then saves the inpainted result.
    """

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

    # Load models
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







































    
