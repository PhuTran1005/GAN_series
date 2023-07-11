import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
import yaml

from model import Generator, Discriminator


### Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3. Different learning rate (is there a better one)
# 4. Change architecture to a CNN


parser = argparse.ArgumentParser()
parser.add_argument("--config_path",
                        type=str,
                        default="./config.yaml",
                        help="Path to train config file.")
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    configs = yaml.full_load(f)

def train():
    # hyper parameters
    device = configs['DEVICE']
    learning_rate = configs['LR']
    z_dim = configs['Z_DIM']
    image_dim = configs['IMAGE_DIM']
    batch_size = configs['BATCH_SIZE']
    num_epochs = configs['NUM_EPOCHS']

    # create the instance of generator and discriminator
    gen = Generator(z_dim=z_dim, img_dim=image_dim).to(device)
    disc = Discriminator(image_dim).to(device)

    # optim
    optim_disc = optim.Adam(disc.parameters(), lr=configs['LR'])
    optim_gen = optim.Adam(gen.parameters(), lr=configs['LR'])
    criterion = nn.BCELoss()
    
    # create random noise
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # dataset
    _transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(root='dataset/', transform=_transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Write log tensorboard
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    for epoch in range(configs['NUM_EPOCHS']):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device) # (B, 784)
            batch_size = real.shape[0]

            ### train Discriminator: maximize log(D(real)) + log(1-D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            # caculate loss of discriminator with input is real
            loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            # caculate loss of discriminator with input is fake
            loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2

            disc.zero_grad()
            loss_D.backward(retain_graph=True)
            optim_disc.step()

            ### train Generator min log(1-D(G(z))) <==> max log(D(G(z)))
            output = disc(fake).view(-1)
            # caculate loss of generator
            loss_G = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_G.backward()
            optim_gen.step()

            # write log when training
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] \ "
                    f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    real = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                    writer_fake.add_image(
                        "MNIST fake images:", img_grid_fake, global_step=step
                    )

                    writer_real.add_image(
                        "MNIST real images:", img_grid_real, global_step=step
                    )

                    step += 1


if __name__ == '__main__':
    train()