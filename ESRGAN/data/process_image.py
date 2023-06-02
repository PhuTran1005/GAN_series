import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

import glob
import numpy as np
from PIL import Image


# Normalize params for pre-trained Pytorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """Denormalizes image tensors using mean and std

    Args:
        tensors (Tensor): Tensor input

    Returns:
        Tensor: Tensor after being denormalized
    """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean(c))
    
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        """Initialization Function

        Args:
            root (str): Path to folder contain image input
            hr_shape (tuple): input size of high resolution image
        """
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
        
        self.files = sorted(glob.blob(root + '/*.*'))
    
    def __getitem__(self, index):
        """Overwrite getitem() method to get item from data input

        Args:
            index (int): index of item that wanting to get

        Returns:
            Tensor: The item that we want to get
        """
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}
    
    def __len__(self):
        """Return len of data input

        Returns:
            int: len of data input
        """
        return len(self.files)
