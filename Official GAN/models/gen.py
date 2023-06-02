import torch
import torch.nn as nn

from torch.autograd import Variable

import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, img_size):
        """Initialization function

        Args:
            latent_dim (int): dimensionality of the latent space
            channels (int): number channels of image
            img_size (int): resolution of input image
        """
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            """Function to create a block include Linear layer, Batch Norm (Option) and activation

                Args:
                    in_feat (int): size of each input sample
                    out_feat (int): size of each output sample
            """
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size
        self.img_shape = (self.channels, self.img_size, self.img_size)
        
        # Define a MLP network
        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        """Forward pass implementation

            Args: 
                z: input noise vector
            Return:
                img(Tensor): generated image
        """
        img = self.model(z) # img shape: torch.Size([16, 784])
        img = img.view(img.size(0), *self.img_shape)

        return img
    
if __name__ == '__main__':
    gen = Generator(latent_dim=100, channels=1, img_size=28)
    print("*"*20 + "Model's Layers" + "*"*20)
    print(gen)
    print("*"*20 + "Model's Layers" + "*"*20)

    # Sample noise as generator input
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    gen = gen.cuda() if cuda else gen

    z = Variable(Tensor(np.random.normal(0, 1, (16, 100))))
    print("z noise shape: ", z.shape) # z noise shape: torch.Size([16, 100])
    output = gen(z)
    print("Output generator shape: ", output.shape) # Output generator shape: torch.Size([16, 1, 28, 28])

