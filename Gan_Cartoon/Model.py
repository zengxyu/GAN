import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

from PIL import Image

import os
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# --------------Start--------- Read images -----------Start

image_size = 32

transform = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.Resize(image_size + 8),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)


class CartoonDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        # Make a list, which includes the image paths and image labels
        file_paths = os.listdir(dir_path)
        image_paths = []
        for file_path in file_paths:
            image_path = os.path.join(dir_path, file_path)
            image_paths.append(image_path)
        print("The size of image datasets : ", len(image_paths))
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, item):
        # return each image item
        image_path = self.image_paths[item]
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        # return the size of images
        return len(self.image_paths)


# --------------End-----------Read images ------------End


# --------------Start--------- Define Model ----------Start
NOISE_DIM = 200
NUM_TRAIN = 5000
batch_size = 64
fake_images_dir = "../Gan_Cartoon/data/fake_images"
cartoon_images_dir = "E:\\Dataset\\anime_face_resize"
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
            nn.Linear(1024, 4 * 4 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 4 * 128)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


G = Generator().to(device)
D = Discriminator().to(device)
bce_loss = nn.BCELoss()
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
dataset = CartoonDataset(dir_path=cartoon_images_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
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
    for i, (images) in enumerate(data_loader):
        if len(images) is not batch_size:
            break
        count = count + 1
        images = images.reshape(batch_size, 3, 32, 32).to(device)

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
            print(
                "epoch: {}, Iter: {}, D:{:.4}, G:{:.4},D(x): {:.2f}, D(G(z)): {:.2f}, D(G(z2)):{:.2f}".format(epoch, i,
                                                                                                              d_loss.data,
                                                                                                              g_loss.data,
                                                                                                              rd_output.mean().item(),
                                                                                                              fd_output.mean().item(),
                                                                                                              fd2_output.mean().item()))
        if i % 200 == 0:
            # Store the images
            fake_images = fake_images.reshape(batch_size, 3, 32, 32)
            save_image(denorm(fake_images),
                       os.path.join(fake_images_dir, 'fake_images-{}-{}.png'.format(epoch + 1, count)))
# --------------End--------- Training ----------End
