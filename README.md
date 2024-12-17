# Denoising Diffusion Probabilistic Model and Image Inpainting

This project aims to implement, train and use a `Denoising Diffusion Probabilistic Model (DDPM)` presented in the paper by [Ho et al., 2020](https://arxiv.org/abs/2006.11239).

In this project, we successfully implemented a Denoising Diffusion Probabilistic Model (DDPM) from scratch, focusing on both the theoretical and practical aspects of the framework. We introduced a **simplified version of the U-Net architecture** to reduce computational complexity while maintaining performance. To enhance the model's effectiveness, we explored different **noise schedulers** and performed **hyperparameter tuning** to optimize the training process.  

Furthermore, we demonstrated the model's capabilities through **sample generation** and **image inpainting tasks**, showcasing its ability to produce high-quality and coherent results. To address the limitations of standard U-Net, we developed an improved version by integrating **attention mechanisms**, enabling the model to better capture long-range dependencies and refine global structures in the images.  

These contributions collectively advance the understanding and implementation of DDPMs, providing a solid foundation for future work on diffusion-based generative models and their applications.


Supports MNIST, Fashion-MNIST, CIFAR10 and CelebA datasets. You can use any other dataset as well, but the model architecture and training parameters might need some changes.


## Content description
Here, we will give a short description about the usecases of every file and folder in this project.

* `Inpaint_images`: contains the original images, masked images, as well as the inpainted images based on the RePaint pipeline implemented in `pipelines.py`. Since our analysis was based on the Linear and the Logarithmic noise scheduler for the Diffusion model (see explanation in the  `Report.pdf`), we tested our 2 trained models based on these 2 schedulers on the RePaint task.

* `diffusion_model`: This folder contains the implementation of the Denoising Diffusion Probabilistic Model (DDPM) from scratch. `models.py` includes the implementation of the Diffusion model (Forward method for noise injection, Sample method for sample generation based on trained model and reverse noise injection), the U-Net architecture with and without Attention blocks as well as the InPaint model to perform image inpainting. `blocks.py` contains the implementation of the different blocks used in the U-Net architecture.

* `generated_samples`: contains an example of sample generation by the trained models based on different types of schedulers and datasets.

* `hyperparam_tuning`: contains the results of the hyperparameter tuning process using Optuna wich uses the Bayesian Optimization process. The results are stored based on the type of noise scheduler implemented, as well as the type of dataset.

* `plots`: contains the training and validation loss plots for the different schedulers.

* `saved_models`: contains the saved trained models for the different schedulers and datasets. These models are outputs of the Hyperparameters tuning pipeline that saves the best models and best hyperparameters for each scheduler and dataset.

* `schedules`:  Contains the implementation of the different schedulers used in the project. Scheduler selection in based on the scheduler tuning pipeline.

* `dataset.py`: contains the implementation of the class DiffSet that manipulates the type of dataset used in the project.

* `main.py`: contains the main function that calls the different functions to train the model, tune best schedulers, tune hyperparameters, generate samples, and perform image inpainting.

* `pipelines.py`: contains the implementation of the different pipeliens used in `main.py` such as the training pipeline, the hyperparameter tuning pipeline, the scheduler tuning pipeline, the sample generation pipeline and the image inpainting pipeline.

* `requirements.txt`: contains the required library to install for the project.

* `tools.py`: contains some useful functions used in the different pipelines to save images, loads datasets, plot losses and save best trained models.

* `train_test.py`: contains the implementation of the train function used to train and validate the different diffusion models, as well as the test function to test the trained model based on test set.


## Requirements

* torch
* torchaudio
* numpy
* matplotlib
* pytorch-fid
* tools
* matplotlib
* einops
* IPython
* gdown
* pytorch-lightning
* pytorch_fid
* optuna
* joblib



## Generated Images

### MNIST

![MNIST Generation](/imgs/mnist.gif)

### Fashion-MNIST

![Fashion MNIST Generation](/imgs/fashion.gif)

### CIFAR

![CIFAR Generation](/imgs/cifar.gif)
