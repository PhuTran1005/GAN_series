import argparse
import numpy as np

from torchvision.utils import save_image

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models.gen import Generator
from models.disc import Discriminator
from data.mnist import MNIST_LOADER
from data.cifar_10 import CIFAR_10


# define a parser that will parse the command line
parser = argparse.ArgumentParser(description="training GAN network...")

# define some argument as variable
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of input image")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")

opt = parser.parse_args()

if __name__ == "__main__":
    # image shape
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    print("Image shape: ", img_shape)

    # check cuda is availabe or not
    cuda = True if torch.cuda.is_available() else False
    print("Using cuda for training - ", cuda)

    # define loss function
    adversarial_loss = torch.nn.BCELoss()

    # initialize generator and discriminator
    gen = Generator(opt.latent_dim, opt.channels, opt.img_size)
    disc = Discriminator(opt.latent_dim, opt.channels, opt.img_size)

    if cuda: # place model and loss function on CUDA if it's availabe
        gen.cuda()
        disc.cuda()
        adversarial_loss.cuda()

    # define optimizer for Generator and Discriminator
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get data
    mnist = MNIST_LOADER(opt.img_size, opt.batch_size)
    train_loader = mnist.mnist_data()

    cifar10 = CIFAR_10(opt.img_size, opt.batch_size)
    train_loader_cifar10, val_loader_cifar10, test_loader_cifar10 = cifar10.load_img()

    print("-*-"*40)
    print("Training Process...")
    print("-*-"*40)

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_loader_cifar10):
            # adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # configure input
            real_imgs = Variable(imgs.type(Tensor))

            print("Training Generator Process...")
            print("-*-"*40)

            optimizer_G.zero_grad()

            # sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # generate a batch of images
            gen_imgs = gen(z)
            # loss measures generators's ability to fool the discriminator
            gen_loss = adversarial_loss(disc(gen_imgs), valid)

            gen_loss.backward()
            optimizer_G.step()

            print("Training Discriminator Process...")
            print("-*-"*40)

            optimizer_D.zero_grad()

            # measure discriminator's ability to classify real from generated images
            real_loss = adversarial_loss(disc(real_imgs), valid)
            fake_loss = adversarial_loss(disc(gen_imgs.detach()), fake)
            disc_loss = (real_loss + fake_loss) / 2

            disc_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_loader), disc_loss.item(), gen_loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)