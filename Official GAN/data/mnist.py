from torchvision import datasets
from torchvision.transforms import transforms

import torch
from torch.utils.data import DataLoader, Dataset

import os


class MNIST_LOADER(Dataset):
    def __init__(self, img_size, batch_size):
        """Initialize function

        Args:
            img_size (int): size of input image
            batch_size (int): size of the batches
        """
        super(MNIST_LOADER, self).__init__()
        
        self.img_size = img_size
        self.batch_size = batch_size
    
    def mnist_data(self):
        """Make train loader of MNIST dataset

        Returns:
            DataLoader: train loader
        """
        # make a directory to store images
        os.makedirs("images", exist_ok=True)

        # check cuda availability
        cuda = True if torch.cuda.is_available() else False

        # configure Data Loader
        os.makedirs("mnist", exist_ok=True)

        # define a train loader
        train_loader = DataLoader(
            datasets.MNIST(
                "mnist",
                train=True,download=True,
                transform = transforms.Compose(
                    [transforms.Resize(self.img_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5])]
                )
            ),
            batch_size = self.batch_size,
            shuffle=True
        )

        return train_loader