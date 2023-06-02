import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        """Initialization Function of Discriminator

        Args:
            input_shape (tuple): intput size of input images
        """
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.channels, self.height, self.width = self.input_shape
        patch_h, patch_w = int(self.height / 2**4), int(self.width / 2**4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            """Implementation Discriminator block

            Args:
                in_filters (int): number of input filters
                out_filters (int): number of output filter
                first_block (bool, optional): Is block first? Defaults to False.

            Returns:
                list: list of layers in one discriminator block
            """
            layers = []
            layers.append(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            )

            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(
                nn.LeakyReLU()
            )
            layers.append(
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            layers.append(
                nn.BatchNorm2d(out_filters)
            )
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            return layers
        
        layers = []
        in_filters = self.channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i==0))
            )
            in_filters = out_filters
        
        layers.append(
            nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        """Forward pass implementation

        Args:
            img (Tensor): input images

        Returns:
            int: Input image is real or fake
        """

        return self.model(img)

