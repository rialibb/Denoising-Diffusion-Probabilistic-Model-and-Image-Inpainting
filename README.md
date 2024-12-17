# Denoising Diffusion Probabilistic Model and Image Inpainting

This project aims to implement, train and use a `Denoising Diffusion Probabilistic Model (DDPM)` presented in the paper by [Ho et al., 2020](https://arxiv.org/abs/2006.11239).

In this project, we successfully implemented a Denoising Diffusion Probabilistic Model (DDPM) from scratch, focusing on both the theoretical and practical aspects of the framework. We introduced a \textbf{simplified version of the U-Net architecture} to reduce computational complexity while maintaining performance. To enhance the model's effectiveness, we explored different \textbf{noise schedulers} and performed \textbf{hyperparameter tuning} to optimize the training process.  

Furthermore, we demonstrated the model's capabilities through \textbf{sample generation} and \textbf{image inpainting tasks}, showcasing its ability to produce high-quality and coherent results. To address the limitations of standard U-Net, we developed an improved version by integrating \textbf{attention mechanisms}, enabling the model to better capture long-range dependencies and refine global structures in the images.  

These contributions collectively advance the understanding and implementation of DDPMs, providing a solid foundation for future work on diffusion-based generative models and their applications.


Supports MNIST, Fashion-MNIST, CIFAR10 and CelebA datasets. You can use any other dataset as well, but the model architecture and training parameters might need some changes.

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


## Content description
Here, we will give a short description about the usecases of every file and folder in this project.

* `Inpaint_images folder` : contains the original images, masked images, as well as the inpainted images based on the RePaint pipeline implemented in `pipelines.py`. Since our analysis was based on the Linear and the Logarithmic noise scheduler for the Diffusion model (see explanation in the  `Report.pdf`), we tested our 2 trained models based on these 2 schedulers on the RePaint task.

* `diffusion Model folder`: 


## Generated Images

### MNIST

![MNIST Generation](/imgs/mnist.gif)

### Fashion-MNIST

![Fashion MNIST Generation](/imgs/fashion.gif)

### CIFAR

![CIFAR Generation](/imgs/cifar.gif)
