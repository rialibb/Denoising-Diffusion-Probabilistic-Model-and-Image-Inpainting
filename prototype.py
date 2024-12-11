       
       
       
       
       
       
               # Validation Phase
        diffusion.eval()
        unet.eval()
        val_loss = []
        
        real_images = []
        generated_images = []

        with torch.no_grad():
            for images in valloader:  
                images = images.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
                epsilon = torch.randn(images.shape, device=device)
                
                # Forward pass
                x_t = diffusion.forward(images, t, epsilon)
                predicted_noise = unet.forward(x_t, time=t)
                
                # Compute loss
                loss = loss_func(predicted_noise, epsilon)
                val_loss.append(loss.item())
                
                # add batch images 
                real_images.append(images.cpu())
                # add generated images
                generated_samples = diffusion.sample(unet, images.shape).cpu()
                generated_images.append(generated_samples)
                
        avg_val_loss = np.mean(val_loss)
       
       
       
       
       
       
       
       
        # Flatten lists into tensors
        real_images = torch.cat(real_images)
        generated_images = torch.cat(generated_images)

        # Normalize images to [0, 1] for FID calculation
        real_images = ((real_images + 1) / 2).clamp(0, 1)
        generated_images = ((generated_images + 1) / 2).clamp(0, 1)
        #calculate bthe FID score
        FID = fid_score.calculate_from_tensors(real_images, generated_images, device=device)