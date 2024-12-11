from einops import rearrange
import matplotlib.pyplot as plt
from IPython import display




def show_images(images, ncol=10, figsize=(8,8), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    if out.shape[0] == 1:
        ax.matshow(out[0], **kwargs)
    else:
        ax.imshow(out.permute((1, 2, 0)), **kwargs)
    ax.set_title('Sample Generation')
    display.display(fig)
    plt.close(fig)


