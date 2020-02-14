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
sample_dir = "data/fake_samples"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
else:
    shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)


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

mnist = torchvision.datasets.MNIST(root='data/mnist',
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
        images = images.reshape(batch_size, 1,28,28).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # sample mini batch of m examples from true distribution
        rd_output = D(images)
        # sample m noise
        noise = ((torch.rand(batch_size, NOISE_DIM) - 0.5) / 0.5).to(device)
        fake_images = G(noise)
        fd_output = D(fake_images)
        d_loss = bce_loss(rd_output, real_labels) + bce_loss(fd_output, fake_labels)
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # sample mini batch of m examples from true distribution
        rd_output = D(images)
        # sample m noise
        noise = ((torch.rand(batch_size, NOISE_DIM) - 0.5) / 0.5).to(device)
        fake_images = G(noise)
        fd2_output = D(fake_images)
        g_loss = bce_loss(fd2_output, real_labels)
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        if (i % 100 == 0):
            print("epoch: {}, Iter: {}, D:{:.4}, G:{:.4},D(x): {:.2f}, D(G(z)): {:.2f}".format(epoch, i, d_loss.data,
                                                                                               g_loss.data,
                                                                                               rd_output.mean().item(),
                                                                                               fd_output.mean().item()))
        if i % 200 == 0:
            # Store the images
            fake_images = fake_images.reshape(batch_size, 1, 28, 28)
            save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}-{}.png'.format(epoch + 1, count)))
# --------------End--------- Training ----------End
