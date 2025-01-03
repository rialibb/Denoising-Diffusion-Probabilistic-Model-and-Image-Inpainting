import torch
from schedules.schedules import select_betas
from diffusion_model import Diffusion, UNet, UNet_with_attention, InPaint
from train_test import f_train, f_test
from config import device
from tools import load_data, save_images, plot_losses, SaveBestModelCallback
import optuna
import os
import joblib
import numpy as np





def run_training_and_testing_pipeline(
    T = 1000,
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 20,
    batch_size = 128,
    lr = 0.001,
    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    beta_min = 0.0001,
    beta_max = 0.02,
    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
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
        model (str): Model name to be used for the reversse process.
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
    if model == "unet_no_attention":
        unet = UNet(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
        comp = "no_attention"
    elif model == "unet_with_attention":
        unet = UNet_with_attention(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
        comp = "with_attention"
    diffusion.to(device)
    unet.to(device)

    # Create the folder if it doesn't exist
    save_dir = os.path.join(save_dir, scheduler)
    os.makedirs(save_dir, exist_ok=True)
    # create the save path for the model
    save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_diffusion_{comp}.pth')
    save_path_unet = os.path.join(save_dir, f'{dataset_choice}_{model}.pth')

    # Train models
    train_losses, val_losses = f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)
    # Test models
    f_test(diffusion, unet, test_loader)
    # Save best models
    torch.save(diffusion.state_dict(), save_path_diffusion)
    torch.save(unet.state_dict(), save_path_unet)









def run_scheduler_tuning_pipeline(
    T = 1000,
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 20,
    batch_size = 128,
    lr = 0.001,
    beta_min = 0.0001,
    beta_max = 0.02,
    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    save_dir: str = 'saved_models'
    ):
    """
    Find the best scheduler for the given dataset and model architecture.

    Args:
        T (int): Total number of time steps for the diffusion process.
        dataset_choice (str): Dataset to use.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for the DataLoaders.
        lr (float): Learning rate for the optimizer.
        beta_min (float): Minimum beta value for the scheduler.
        beta_max (float): Maximum beta value for the scheduler.
        model (str): Model name to be used for the reversse process.
        save_dir (str): Directory to save the trained models.

    The function performs the following steps:
    1. Loads the specified dataset and splits it into train, validation, and test sets.
    2. Initializes the scheduler, diffusion model, and UNet model.
    3. Trains the models using the training and validation sets based on different schedulers
    4. Tests the best model using the test set.
    5. Saves the best trained model to the specified directory.
    6. Plots training and validation loss across epochs.
    """

    # define the different schedulers :
    schedulers = ['linear', 'cosine', 'quadratic', 'exponential', 'logarithmic']
    # Load data
    train_loader, val_loader, test_loader, test_dataset = load_data(dataset_choice, batch_size)
    
    # initialize the losses
    all_train_losses, all_val_losses = [], []

    # train models with different schedulers
    best_val_losses = []
    diffusions =[]
    unets = []
    
    for scheduler in schedulers:
        print(f'Training models with {scheduler} scheduler...')
        # select scheduler
        betas = select_betas(scheduler, beta_min, beta_max, T)

        # Create models
        diffusion = Diffusion(betas, T)
        if model == "unet_no_attention":
            unet = UNet(
                img_channels = test_dataset.depth,
                base_channels = test_dataset.size,
                time_emb_dim = test_dataset.size,
                num_classes = None,
            )
            comp = "no_attention"
        elif model == "unet_with_attention":
            unet = UNet_with_attention(
                img_channels = test_dataset.depth,
                base_channels = test_dataset.size,
                time_emb_dim = test_dataset.size,
                num_classes = None,
            )
            comp = "with_attention"
        diffusion.to(device)
        unet.to(device)
        
        # Train models
        train_losses, val_losses = f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)
        # add losses
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        best_val_losses.append(val_losses[-1])
        # add models
        diffusions.append(diffusion)
        unets.append(unet)
        
    #plot different losses
    plot_losses(all_train_losses, all_val_losses, schedulers)

    #find best scheduler
    best_index = np.argmin(best_val_losses)
    best_scheduler = schedulers[best_index]
    best_diffusion = diffusions[best_index]
    best_unet = unets[best_index]
    
    print(f'________ Best scheduler: {best_scheduler}__________')
    print(f'________ Best validation loss: {best_val_losses[best_index]}__________')
    
    # Test models
    f_test(best_diffusion, best_unet, test_loader)
    # Create the folder if it doesn't exist
    save_dir = os.path.join(save_dir, best_scheduler)
    os.makedirs(save_dir, exist_ok=True)
    # create the save path for the model
    save_path_diffusion = os.path.join(save_dir, f'{dataset_choice}_diffusion_{comp}.pth')
    save_path_unet = os.path.join(save_dir, f'{dataset_choice}_{model}.pth')
    # Save best models
    torch.save(best_diffusion.state_dict(), save_path_diffusion)
    torch.save(best_unet.state_dict(), save_path_unet)

    
    
    
    
    
    
    




def run_hyperparam_tuning_pipeline(
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    n_epochs = 15,
    batch_size = 128,
    num_trials = 5,
    lr = 0.001,
    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    ):
    """
    Runs hyperparameter tuning using Bayesian Optimization, with Optuna.

    Args:
        dataset_choice (str): Dataset to use.
        n_epochs (int): Number of epochs for training in each trial.
        batch_size (int): Batch size for the DataLoaders.
        num_trials (int): Number of trials to run for hyperparameter tuning.
        lr (float) : Learning rate for the optimizer.
        scheduler (str): Beta scheduler.
        model (str): Model name to be used for the reversse process.

    The function performs the following steps:
    1. Loads the dataset and splits it into train, validation, and test sets.
    2. Uses Optuna to optimize hyperparameters for the learning rate, beta_min, and beta_max.
    3. Trains a diffusion model and a UNet for each trial.
    4. Tests the model and saves the best-performing models.
    5. Saves the Optuna study results to a file for future reference.
    """

    # Load datasets
    train_loader, val_loader, _, test_dataset = load_data(dataset_choice, batch_size)
    
    # initialize the callback function
    save_best_model = SaveBestModelCallback()

    # Objective function for Optuna
    def objective(trial):

        # Hyperparameters to tune
        T = trial.suggest_int('TimeSteps', 500, 2000)
        beta_min = trial.suggest_float('beta_min', 1e-5, 1e-3, log=True)
        beta_max = trial.suggest_float('beta_max', 0.01, 0.05)
        # Lookup table to set scheduler
        betas = select_betas(scheduler, beta_min, beta_max, T)
        
        # Create diffusion and UNet models
        diffusion = Diffusion(betas, num_timesteps=T)
        if model == "unet_no_attention":
            unet = UNet(
                img_channels = test_dataset.depth,
                base_channels = test_dataset.size,
                time_emb_dim = test_dataset.size,
                num_classes = None,
            )
        elif model == "unet_with_attention":
            unet = UNet_with_attention(
                img_channels = test_dataset.depth,
                base_channels = test_dataset.size,
                time_emb_dim = test_dataset.size,
                num_classes = None,
            )
        diffusion.to(device)
        unet.to(device)

        # Train the model
        _, val_losses = f_train(diffusion, unet, train_loader, val_loader, n_epochs=n_epochs, learning_rate=lr)

        # Save only best model
        save_best_model(diffusion, unet, val_losses[-1], dataset_choice, scheduler, comp, model)
    
        return val_losses[-1]
    
    if model == "unet_with_attention":
        comp = "with_attention"
    else:
        comp = "no_attention"

    # Set up the Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    print("Best trial:")
    print(f"  Value (Validation Loss): {study.best_trial.value}")
    print(f"  Hyperparameters: {study.best_trial.params}")

    # Save the study results in a file for later use
    os.makedirs(f'hyperparam_tuning/{scheduler}', exist_ok=True)
    joblib.dump(study, f'hyperparam_tuning/{scheduler}/{dataset_choice}_{comp}_optuna_study.pkl')






def run_sampling_pipeline(
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    batch_size = 128,
    scheduler =  "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    ):
    """
    Runs the sampling pipeline using a diffusion model and UNet.

    Args:
        dataset_choice (str): Dataset to use.
        batch_size (int): Batch size for loading the dataset.
        scheduler (str): Beta scheduler.
        model (str): Model name to be used for the reversse process.

    The function performs the following steps:
    1. Loads the test dataset for the specified dataset choice.
    2. Initializes and loads pre-trained weights for the diffusion and UNet models.
    3. Generates and saves a batch of sample images from the diffusion model.
    """
    # load best hyperparameters
    comp = "no_attention" if model == "unet_no_attention" else "with_attention"
    study_path = f'hyperparam_tuning/{scheduler}/{dataset_choice}_{comp}_optuna_study.pkl'
    
    # Check if the file exists
    if not os.path.exists(study_path):
        raise FileNotFoundError(f"Optuna study file not found: {study_path}")
    
    study = joblib.load(study_path)
    
    best_params = study.best_trial.params

    # Assign parameters to variables
    T = best_params['TimeSteps']
    beta_min = best_params['beta_min']
    beta_max = best_params['beta_max']

    _, _, _, test_dataset = load_data(dataset_choice, batch_size)

    betas = select_betas(scheduler, beta_min, beta_max, T)

    # Create models
    diffusion = Diffusion(betas, T)
    if model == "unet_no_attention":
        unet = UNet(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
    elif model == "unet_with_attention":
        unet = UNet_with_attention(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
    diffusion.to(device)
    unet.to(device)

    # Load models
    diffusion.load_state_dict(torch.load(f'saved_models/{scheduler}/{dataset_choice}_best_diffusion_{comp}.pth'))
    unet.load_state_dict(torch.load(f'saved_models/{scheduler}/{dataset_choice}_best_{model}.pth'))
    
    # Sample generation
    x_shape = (25, test_dataset.depth, test_dataset.size, test_dataset.size)
    samples = diffusion.sample(unet, x_shape)
    samples01 = ((samples + 1) / 2).clip(0, 1)
    save_images(samples01, dataset_choice, save_dir=f'generated_samples/{scheduler}', image_type = 'samples' , cmap='binary', comp =comp, ncol=5)
    
    
    
    






def run_inpainting_pipeline(
    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    batch_size = 128,
    scheduler =  "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    image_index = 5643,
    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    ):
    """
    Runs the inpainting pipeline using a diffusion model and UNet.

    Args:
        dataset_choice (str): Dataset to use.
        batch_size (int): Batch size for loading the dataset.
        scheduler (str): Beta scheduler.
        image_index (int): an index of an image from test set to inpaint.
        model (str): Model name to be used for the reversse process.

    The function performs the following steps:
    1. Loads the test dataset for the specified dataset choice.
    2. Initializes and loads pre-trained weights for the diffusion and UNet models.
    3. Selects a single test image, performs image inpainting and then saves the inpainted result.
    """
    # load best hyperparameters
    comp = "no_attention" if model == "unet_no_attention" else "with_attention"
    study_path = f'hyperparam_tuning/{scheduler}/{dataset_choice}_{comp}_optuna_study.pkl'
    
    # Check if the file exists
    if not os.path.exists(study_path):
        raise FileNotFoundError(f"Optuna study file not found: {study_path}")
    
    study = joblib.load(study_path)
    
    best_params = study.best_trial.params

    # Assign parameters to variables
    T = best_params['TimeSteps']
    beta_min = best_params['beta_min']
    beta_max = best_params['beta_max']

    _, _, _, test_dataset = load_data(dataset_choice, batch_size)

    betas = select_betas(scheduler, beta_min, beta_max, T)

    # Create models
    diffusion = Diffusion(betas, T)
    if model == "unet_no_attention":
        unet = UNet(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
    elif model == "unet_with_attention":
        unet = UNet_with_attention(
            img_channels = test_dataset.depth,
            base_channels = test_dataset.size,
            time_emb_dim = test_dataset.size,
            num_classes = None,
        )
    diffusion.to(device)
    unet.to(device)

    # Load models
    diffusion.load_state_dict(torch.load(f'saved_models/{scheduler}/{dataset_choice}_best_diffusion_{comp}.pth'))
    unet.load_state_dict(torch.load(f'saved_models/{scheduler}/{dataset_choice}_best_{model}.pth'))

    # Inpainting masked image
    image = test_dataset[image_index]  # Select one image from the test dataset
    inpaint = InPaint()
    inpaint.sample(diffusion, unet, image, dataset_choice, scheduler, comp)