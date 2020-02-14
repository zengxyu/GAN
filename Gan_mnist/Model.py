import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# --------------Start--------- Define Model ----------Start
NOISE_DIM = 96
NUM_TRAIN = 50
batch_size = 32
fake_images_dir = "../Gan_mnist/data/fake_images"
mnist_images_dir = "../Gan_mnist/data/mnist"
if not os.path.exists(fake_images_dir):
    os.makedirs(fake_images_dir)
else:
    shutil.rmtree(fake_images_dir)
    os.makedirs(fake_images_dir)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


G = Generator().to(device)
D = Discriminator().to(device)
bce_loss = nn.BCEWithLogitsLoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


# --------------End--------- Define Model ----------End


# --------------Start--------- Load Data ----------Start
# Image processing: normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))])

mnist = torchvision.datasets.MNIST(root=mnist_images_dir,
                                   train=True,
                                   transform=transform,
                                   download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# --------------End--------- Load Data ----------End


# --------------Start--------- Training ----------Start
# Following code is for training
# Thought:
# Fix Generator, optimize Discriminator
# Fix Discriminator, optimize Generator
for epoch in range(NUM_TRAIN):
    total_step = len(data_loader)
    count = 0
    for i, (images, _) in enumerate(data_loader):
        count = count + 1
        images = images.reshape(batch_size, 1, 28, 28).to(device)

        # ---------------Train Discriminator -------start------------#
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # sample mini batch of m examples from true distribution,
        # feed real images into discriminator, output the labels of real images
        rd_output = D(images)
        # sample m noise
        noise = ((torch.rand(batch_size, NOISE_DIM) - 0.5) / 0.5).to(device)
        # Feed the noise into the generator, output fake images
        fake_images = G(noise)
        # Feed fake images into discriminator, output labels of fake images
        fd_output = D(fake_images)
        # Compute the loss
        d_loss = bce_loss(rd_output, real_labels) + bce_loss(fd_output, fake_labels)
        # Back propagation and update the parameters
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # ---------------Train discriminator ---------end----------#

        # ---------------Train generator ---------start----------#
        # sample m noise
        noise = ((torch.rand(batch_size, NOISE_DIM) - 0.5) / 0.5).to(device)
        # Feed the noise into the generator, output fake images
        fake_images = G(noise)
        # Feed fake images into discriminator, output labels of fake images
        fd2_output = D(fake_images)
        # Compute the loss
        g_loss = bce_loss(fd2_output, real_labels)
        # Back propagation
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # ---------------Train generator --------end-----------#
        # Print training information
        if (i % 100 == 0):
            print("epoch: {}, Iter: {}, D:{:.4}, G:{:.4},D(x): {:.2f}, D(G(z)): {:.2f}".format(epoch, i, d_loss.data,
                                                                                               g_loss.data,
                                                                                               rd_output.mean().item(),
                                                                                               fd_output.mean().item()))
        if i % 200 == 0:
            # Store the images
            fake_images = fake_images.reshape(batch_size, 1, 28, 28)
            save_image(denorm(fake_images),
                       os.path.join(fake_images_dir, 'fake_images-{}-{}.png'.format(epoch + 1, count)))
# --------------End--------- Training ----------End
