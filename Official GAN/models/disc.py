import torch
import torch.nn as nn

import numpy as np


class Discriminator(nn.Module):
    def __init__(self, latent_dim, channels, img_size):
        """Initialize function

        Args:
            latent_dim (int): dimensionality of the latent space
            channels (int): number channels of image
            img_size (int): resolution of input image
        """
        super(Discriminator, self).__init__()

        self.channels = channels
        self.img_size = img_size
        self.img_shape = (self.channels, self.img_size, self.img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """forward pass implementation

        Args:
            img (Tensor): input image

        Returns:
            _type_: _description_
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    

if __name__ == '__main__':
    disc = Discriminator(latent_dim=100, channels=1, img_size=28)
    print("*"*20 + "Model's Layers" + "*"*20)
    print(disc)
    print("*"*20 + "Model's Layers" + "*"*20)

    gen_img = torch.rand([16, 28, 28])
    print("gen_img shape: ", gen_img.shape)
    output = disc(gen_img)
    print("output of disciminator shape: ", output.shape)