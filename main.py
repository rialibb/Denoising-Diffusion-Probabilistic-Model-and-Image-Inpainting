from pipelines import (run_training_and_testing_pipeline,
                       run_scheduler_tuning_pipeline,
                       run_hyperparam_tuning_pipeline, 
                       run_sampling_pipeline,
                       run_inpainting_pipeline)



if __name__ == "__main__":
    #
    # Choose the pipeline you want to run and comment the rest 
    #
    
    #run_training_and_testing_pipeline(
    #    T = 1000,
    #    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    #    n_epochs = 20,
    #    batch_size = 128,
    #    lr = 0.001,
    #    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    #    beta_min = 0.0001,
    #    beta_max = 0.02,
    #    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    #)

    #run_scheduler_tuning_pipeline(
    #    T = 1000,
    #    dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    #    n_epochs = 20,
    #    batch_size = 128,
    #    lr = 0.001,
    #    beta_min = 0.0001,
    #    beta_max = 0.02,
    #    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    #)
    
    #run_hyperparam_tuning_pipeline(
    #    dataset_choice = "CIFAR",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    #    n_epochs = 15,
    #    batch_size = 128,
    #    num_trials = 5,
    #    lr = 0.001,
    #    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    #    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    #)

    #run_sampling_pipeline(
    #    dataset_choice = "CIFAR",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
    #    batch_size = 128,
    #    scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    #    model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    #)
    
    run_inpainting_pipeline(
        dataset_choice = "CIFAR",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
        batch_size = 128,
        scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
        image_index = 456,
        model = "unet_no_attention", # "unet_no_attention", "unet_with_attention"
    )