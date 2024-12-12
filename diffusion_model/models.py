import torch
import torch.nn as nn
import pytorch_lightning as pl
from .blocks import ResidualBlock, Downsample, Upsample, PositionalEmbedding
from config import device
import numpy as np
from tools import save_images


class Diffusion(nn.Module):
    """Diffusion model with a linear schedule of the temperatures.
    """
    def __init__(self, betas, num_timesteps=1000):
        super().__init__()
        
        self.num_timesteps=num_timesteps
        self.betas = betas
        
        
                
    def forward(self, x, t, noise=None):
        """
        Args:
          x of shape (batch_size, ...): Input samples.
          t of shape (batch_size,): Corruption temperatures.
          noise of shape (batch_size, ...): Noise instanses used for corruption.
        """
        
        a=self.alpha_bar(t,self.betas, x.shape)
        x_t = torch.sqrt(a) * x + torch.sqrt(1-a) * self.f(noise)
        return x_t
        
        

    @torch.no_grad()
    def sample(self, model, x_shape, labels=None):
        """
        Args:
          model: A denoising model. model(x, t, labels) takes as inputs:
                   x of shape (batch_size, n_channels, H, W): corrupted examples.
                   t of shape (batch_size,): LongTensor of time steps.
                   labels of shape (batch_size,): LongTensor of the classes of the examples in x.
                 and outputs a denoised version of input x.
          x_shape: The shape of the generated data. For example, to generate batch_size images of shape (1, H, W),
                   x_shape should be (batch_size, 1, H, W).
          labels of shape (batch_size,): LongTensor of the classes of generated samples. None for no conditioning
                   on classes.
        
        Note: Create new tensors on the same device where the model is.
        """

        x = torch.randn(x_shape).to(device)
        for t in range(self.num_timesteps, 0, -1):
            if t>1 : 
                z=torch.randn(x_shape).to(device)
            else:
                z=0
            Ti = torch.ones(x_shape[0],device=device)*t
            x=(1/(torch.sqrt(1-self.betas[t-1]))) * (x - (self.betas[t-1])*model(x,Ti,labels) / torch.sqrt(1-torch.prod(1-self.betas[:t])*torch.ones(x_shape[1:]).to(device)))+torch.sqrt(self.betas[t-1])*z
        return x
       
        
    def alpha_bar(self, t, betas,shape):
        """
        Args:
          t of shape (batch_size,): Corruption temperatures.
          betas of shape (num_timesteps,): corruption rate
          shape : the shape of the batch [batch_size, C, H, W]
        Return: 
          a matrix correspond to the value of alpha bar for every observaion in the batch
        """
        a=torch.cumprod(1-betas,dim=0).to(device)
        L = torch.zeros(shape).to(device)
        for i in range (shape[0]):
            L[i] = a[t[i].long()]*torch.ones(shape[1:]).to(device)
        return L


    def f(self, x):
        """
        Args:
          x : noise
        Return: 
          0 is there is no cosidered noise, the noise otherwise
        """
        if x is None :
            return 0
        else :
            return x







class UNet(nn.Module):
    """The denoising model.
    
    Args:
      img_channels (int): Number of image channels.
      base_channels (int): Number of base channels.
      time_emb_dim (int or None): The size of the embedding vector produced by the MLP which embeds the time input.
      num_classes (int or None): Number of classes, None for no conditioning on classes.
    """

    def __init__(self, img_channels, base_channels, time_emb_dim=None, num_classes=None):
        
        super().__init__()
        
        self.mlp=nn.Sequential(
                            PositionalEmbedding(base_channels),
                            nn.Linear(base_channels,time_emb_dim),
                            nn.SiLU(),
                            nn.Linear(time_emb_dim,time_emb_dim))
        
        
        self.encoder1=nn.Conv2d(img_channels,base_channels,3,padding=1)
        self.encoder2=ResidualBlock(base_channels, base_channels, time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.encoder3=Downsample(base_channels)
        self.encoder4=ResidualBlock(base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.encoder5=Downsample(2*base_channels)
        self.encoder6=ResidualBlock(2*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.encoder7=Downsample(2*base_channels)
        self.encoder8=ResidualBlock(2*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        
        
        self.bottleneck=ResidualBlock(2*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        
        
        self.decoder1=ResidualBlock(4*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.decoder2=ResidualBlock(4*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.up1=Upsample(2*base_channels)
        self.decoder3=ResidualBlock(4*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.decoder4=ResidualBlock(4*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.up2=Upsample(2*base_channels)
        self.decoder5=ResidualBlock(4*base_channels,2*base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.decoder6=ResidualBlock(3*base_channels,base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.up3=Upsample(base_channels)
        self.decoder7=ResidualBlock(2*base_channels,base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)
        self.decoder8=ResidualBlock(2*base_channels,base_channels,time_emb_dim=time_emb_dim, num_classes=num_classes, dropout=0.1)                         
        self.decoder_cnv=nn.Conv2d(base_channels,img_channels,3,padding=1)
        
    
    def forward(self, x, time=None, labels=None):
        """Estimate noise instances used to produced corrupted examples `x` with the corruption level determined
        by `time`. `labels` contains the class information of the examples in `x`.

        Args:
          x of shape (batch_size, n_channels, H, W): Corrupted examples.
          time of shape (batch_size,): LongTensor of time steps which determine the corruption level for
                                       the examples in x.
          labels of shape (batch_size,): LongTensor of the classes of the examples in x.
        
        Returns:
          out of shape (batch_size, n_channels, H, W)
        """

        
        t=self.mlp(time)
        
        
        x1=self.encoder1(x)
        x2=self.encoder2(x1,t,labels)
        x3=self.encoder3(x2)
        x4=self.encoder4(x3,t,labels)
        x5=self.encoder5(x4)
        x6=self.encoder6(x5,t,labels)
        x7=self.encoder7(x6)
        x8=self.encoder8(x7,t,labels)
        
        x=self.bottleneck(x8,t,labels)
        
        
        x=self.decoder1(torch.cat((x,x8),dim=1),t,labels)#
        x=self.decoder2(torch.cat((x,x7),dim=1),t,labels)#
        x=self.up1(x)
        x=self.decoder3(torch.cat((x,x6),dim=1),t,labels)#
        x=self.decoder4(torch.cat((x,x5),dim=1),t,labels)#
        x=self.up2(x)
        x=self.decoder5(torch.cat((x,x4),dim=1),t,labels)#
        x=self.decoder6(torch.cat((x,x3),dim=1),t,labels)#
        x=self.up3(x)
        x=self.decoder7(torch.cat((x,x2),dim=1),t,labels)#
        x=self.decoder8(torch.cat((x,x1),dim=1),t,labels)#
        x=self.decoder_cnv(x)
        
        return x
      
      
      
      
      
      
  
  
class InPaint(nn.Module):
    """The inpaint model.
    
    Args:
      img_channels (int): Number of image channels.
      base_channels (int): Number of base channels.
      time_emb_dim (int or None): The size of the embedding vector produced by the MLP which embeds the time input.
      num_classes (int or None): Number of classes, None for no conditioning on classes.
    """

    def __init__(self):  
        super().__init__()
      
      
    def forward(self, diffusion, model, images, mask_known, labels=None):
        """Generate samples conditioned on known parts of images.
        
        Args:
          diffusion (Diffusion): The descriptor of a diffusion model.
          model: A denoising model: model(x, t, labels) outputs a denoised version of input x.
          images of shape (batch_size, n_channels, H, W): Conditioning images.
          mask_known of shape (batch_size, 1, H, W): BoolTensor which specifies known pixels in images (marked as True).
          labels of shape (batch_size,): Classes of images, None for no conditioning on classes.
        
        Returns:
          x of shape (batch_size, n_channels, H, W): Generated samples (one sample per input image).
        """
        # YOUR CODE HERE
        
        
        x_shape = images.shape
        x_t = torch.randn(x_shape).to(device)     
        
        with torch.no_grad() :
          
            for t in range(diffusion.num_timesteps,0,-1):

                # define epsilon here
                if t>1 :
                    epsilon=torch.randn(x_shape).to(device) 
                else :
                    epsilon=0
                # Update known pixels with noise
                alpha=torch.prod(1-diffusion.betas[:t])*torch.ones(x_shape[1:])
                x_t_minus1_known = torch.sqrt(alpha)*images+(1-alpha)*epsilon

                # define z here
                if t>1 :
                    z=torch.randn(x_shape).to(device)  
                else :
                    z=0    
                
                Ti=torch.ones(x_shape[0],device=device)*t
                #update unkown pixels with
                x_t_minus1_unknown = (1/(torch.sqrt(1-diffusion.betas[t-1]))) * (x_t - (diffusion.betas[t-1])*model(x_t,Ti) / torch.sqrt(1-alpha))+torch.sqrt(diffusion.betas[t-1])*z
                
                x_t = np.dot(x_t_minus1_known , mask_known) + np.dot(x_t_minus1_unknown, (~mask_known))

        # Returnx_0
        return x_t    
    
    
    def sample(self, diffusion, model, image, dataset_choice):
      
        images = image[None, 0].tile(100, 1, 1, 1)  # Copy the image to generate multiple samples
        images = images.to(device)
        (batch_size, C, H, W) = images.shape
        samples0 = ((images + 1) / 2).clip(0, 1)
        save_images(samples0, dataset_choice, save_dir='Inpaint_images', image_type = 'original_image' , cmap='binary', ncol=10)

        # mask out the bottom part of every image
        mask_known = torch.zeros(batch_size, C, H, W, dtype=torch.bool, device=device)
        mask_known[:, :, :H//2, :] = 1
        images_known = images * mask_known

        samples1 = ((images_known + 1) / 2).clip(0, 1)
        save_images(samples1, dataset_choice, save_dir='Inpaint_images', image_type = 'masked_image' , cmap='binary', ncol=10)

        samples = self.forward(diffusion, model, images_known, mask_known, labels=None)
        samples2 = ((samples + 1) / 2).clip(0, 1)
        save_images(samples2, dataset_choice, save_dir='Inpaint_images', image_type = 'inpaint_image_image' , cmap='binary', ncol=10)
        