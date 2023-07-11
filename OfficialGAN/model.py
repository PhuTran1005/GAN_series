import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_features):
        """Discriminator implementation

        Args:
            in_features (int): dimension of input feature
        """
        super(Discriminator, self).__init__()

        self.in_features = in_features
        self.disc = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """forward function implementation

        Args:
            x (tensor): input tensor
        """

        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        """Generator implementation

        Args:
            z_dim (tensor): dim input noise (latent noise)
            img_dim (tensor): dim input image
        """
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.img_dim = img_dim
        self.gen = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, self.img_dim),
            nn.Tanh(), # normalize output to range [-1, 1]
        )

    def forward(self, x):
        """forward function implementation

        Args:
            x (tensor): input noise tensor
        """

        return self.gen(x)

