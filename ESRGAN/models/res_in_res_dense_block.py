import torch
import torch.nn as nn

from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        """Using VGG19 as Feature Extractor before activation to improve perceptual loss
        """
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(
            *list(vgg19_model.features.children())[:35]
        )

        def forward(self, img):
            """Forward pass implementation

            Args:
                img (Tensor): Tensor of input image

            Returns:
                Tensor: Output features
            """
            return self.vgg19_54(img)
        
class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        """The core module of ESRGAN (Residual Dense Network for Image Super-Resolution)

        Args:
            filters (int): number of filters in convolution layer
            res_scale (float, optional): _description_. Defaults to 0.2.
        """
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            """Implementation a sub-block include Conv2D and LeakyReLU activation

            Args:
                in_features (int): number of filters in convolution layer
                non_linearity (bool, optional): _description_. Defaults to True.

            Returns:
                Sequential: Sequential layers
            """
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            
            return nn.Sequential(*layers)
        
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        """Forward pass implementation

        Args:
            x (_type_): _description_
        """
        input = x
        for block in self.blocks:
            out_block = block(input)
            input = torch.cat([input, out_block], 1)
        
        return out_block.mul(self.res_scale) + x
    
class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        """Implementation Residual In Residual Dense Block

        Args:
            filters (int): number of filters in convolution layer
            res_scale (float, optional): _description_. Defaults to 0.2.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_block = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )
    
    def forward(self, x):
        """Forward pass implementation

        Args:
            x (Tensor): Feature map input block

        Returns:
            Tensor: Feature map output block
        """

        return self.dense_block(x).mul(self.res_scale) + x
