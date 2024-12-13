from pipelines import (run_training_and_testing_pipeline,
                       run_hyperparam_tuning_pipeline, 
                       run_inpainting_pipeline)








if __name__ == "__main__":

    run_training_and_testing_pipeline(
        T = 1000,
        dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
        n_epochs = 20,
        batch_size = 128,
        lr = 0.001,
        scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
        beta_min = 0.0001,
        beta_max = 0.02
    )

    run_hyperparam_tuning_pipeline(
        T= 1000,
        dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
        n_epochs = 15,
        batch_size = 128,
        num_trials = 5,
        scheduler = "linear",  # "linear", "cosine", "quadratic", "exponential", "logarithmic"
    )

    run_inpainting_pipeline(
        T = 1000,
        dataset_choice = "MNIST",   # "MNIST", "Fashion" ,  "CIFAR" or "CelebA"
        batch_size = 128,
        beta_min = 0.0001,
        beta_max = 0.02,
    )