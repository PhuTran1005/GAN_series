import torch
import torch.nn as nn

from res_in_res_dense_block import ResidualInResidualDenseBlock


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_block=16, num_upsampling=2):
        """Initialization Function of GeneratorRRDB

        Args:
            channels (int): channels of input images
            filters (int, optional): number of filters. Defaults to 64.
            num_res_block (int, optional): number of RRDB block. Defaults to 16.
            num_upsampling (int, optional): number of upsampling. Defaults to 2.
        """
        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_block)]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsampling):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        """Forward pass implementation

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: Output of architecture
        """
        out_conv1 = self.conv1(x)
        out = self.res_blocks(out_conv1)
        out_conv2 = self.conv2(out)
        out = torch.add(out_conv1, out_conv2)
        out = self.upsampling(out)
        out = self.upsampling(out)

        return out