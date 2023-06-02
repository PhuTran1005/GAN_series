from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class CIFAR_10(Dataset):
    def __init__(self, size_img, batch_size):
        """Initialization Function

        Args:
            size_img (int): size of input images
            batch_size (int): size of the batches
        """
        super(CIFAR_10, self)
        
        self.transforms = transforms.Compose(
            [transforms.Resize(size_img),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )
        self.size_img = size_img
        self.batch_size = batch_size

        # download CIFAR10 dataset
        self.dataset = CIFAR10(root="./data", download=True, transform=self.transforms)
        self.test_dataset = CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print classes in CIFAR10
        classes = self.dataset.classes
        print(classes)

    def load_img(self):
        """Load images into DataLoader
        """
        # split dataset into training set and validation set
        torch.manual_seed(42)
        validation_size = 5000
        train_size = len(self.dataset) - validation_size

        train_dataset, val_dataset = random_split(self.dataset, [train_size, validation_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        return train_loader, val_loader, test_loader
    
    def visualize_img(self):
        """Visualize image of CIFAR10 dataset
        """
        # print(self.dataset[0][0].shape)
        fig = plt.figure(figsize=(1,1))
        rows = cols =2
        x_labels = ["x_labels", "(a)", "(b)", "(c)", "(d)"]

        # Solution 1
        # for i in range(4):
        #     ax = fig.add_subplot(rows, cols, i+1)
        #     transform_img = transforms.ToPILImage()
        #     img = transform_img(self.dataset[i][0])
        #     print(img.size)
        #     ax.imshow(img)
        #     ax.set_xlabel(x_labels[i+1])
        #     ax.set_xticks([]), ax.set_yticks([])
        # plt.show()

        # Solution 2
        # Plot some training images
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, pin_memory=True)
        real_batch = next(iter(test_loader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

if __name__ == '__main__':
    cifar10 = CIFAR_10(size_img=32, batch_size=64)
    cifar10.visualize_img()