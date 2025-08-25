import numpy as np
import torch
import crypten
import crypten.nn as nn
import crypten.mpc as mpc
import crypten.communicator as comm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle as pkl
import time

epochs = 3
# We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
n_train_items = 640
n_test_items = 640

# Discriminator hyperparams

# Size of input image to discriminator (28*28)
input_size = 784
# Size of discriminator output (real or fake)
d_output_size = 1
# Size of last hidden layer in the discriminator
d_hidden_size = 32

# Generator hyperparams

# Size of latent vector to give to generator
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 784
# Size of first hidden layer in the generator
g_hidden_size = 32

import crypten
import torch
import torchvision
from crypten import mpc

crypten.init()
torch.set_num_threads(1)
# Alice is party 0
ALICE = 0
BOB = 1

def take_samples(digits, n_samples=60000):
    """Returns images and labels based on sample size"""
    images, labels = [], []

    for i, digit in enumerate(digits):
        if i == n_samples:
            break
        image, label = digit
        images.append(image)
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), 10)
        labels.append(label_one_hot)

    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, labels

@mpc.run_multiprocess(world_size=2)
def save_digits(images):
    sample_alice = images[ :2000]
    sample_bob = images[2000:4000]
    print("sample_alice.size():", sample_alice.size())
    print("sample_bob.size():", sample_bob.size())

    crypten.save_from_party(sample_alice, "/tmp/data/alice_images.pth", src=ALICE)
    crypten.save_from_party(sample_bob, "/tmp/data/bob_images.pth", src=BOB)      


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
        
    # numerically stable loss
    criterion = crypten.nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = crypten.nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

class Generator(crypten.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decon1 = crypten.nn.ConvTranspose2d(100, 256, 4, stride=1, padding=0,bias=False)
        self.bn1 = crypten.nn.BatchNorm2d(256)
        self.decon2 = crypten.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)  
        self.bn2 = crypten.nn.BatchNorm2d(128)
        self.decon3 = crypten.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)  
        self.bn3 = crypten.nn.BatchNorm2d(64)
        self.decon4 = crypten.nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False)
        self.relu = crypten.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.decon1(x))
        x = self.bn1(x)
        x = self.relu(self.decon2(x))
        x = self.bn2(x)
        x = self.relu(self.decon3(x))
        x = self.bn3(x)
        x = self.decon4(x)
        x = x.tanh()
        return x
    

class Discriminator(crypten.nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc 
        self.ndf = ndf
        self.conv1 = crypten.nn.Conv2d(self.nc, self.ndf, 4, stride=2, padding=1, bias=False)  # (64, 14, 14)
        self.leaky_relu = crypten.nn.LeakyReLU(0.2)
        self.conv2 = crypten.nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=1, bias=False)  # (128, 7, 7)
        self.bn1 = crypten.nn.BatchNorm2d(self.ndf * 2)
        self.conv3 = crypten.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=1, bias=False)  # (256, 3, 3)
        self.bn2 = crypten.nn.BatchNorm2d(self.ndf * 4)
        self.conv4 = crypten.nn.Conv2d(self.ndf * 4, 1, 3, stride=1, padding=0, bias=False)  # (1, 1, 1)
        self.sd = crypten.nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))

        x = self.leaky_relu(self.bn1(self.conv2(x)))

        x = self.leaky_relu(self.bn2(self.conv3(x)))

        x = self.sd(self.conv4(x))
        return x.view(-1, 1).squeeze(1)

@mpc.run_multiprocess(world_size=2)
def run_lpgan():
    train_data = torchvision.datasets.MNIST(root='/tmp/data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    num_samples = len(train_data)
    print(f"Number of samples in the training dataset: {num_samples}")

    # 获取第一个样本
    sample_index = 0
    image, label = train_data[sample_index]

    # 转换为 NumPy 数组
    image_np = image.numpy()

    # 显示图像
    plt.imshow(image_np.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    images, labels = take_samples(train_data, n_samples=30000)
    save_digits(images)

    # Example usage
    input_size = (10, 100, 1, 1)
    x = crypten.cryptensor(torch.randn(input_size))
    G = Generator()
    G.encrypt()
    outputG = G(x)

    input_size = (10, 1, 28, 28)
    x = crypten.cryptensor(torch.randn(input_size))
    D = Discriminator(nc=1, ndf=32)
    D.encrypt()
    outputD = D(x)

    # load data
    x_alice_enc = crypten.load_from_party('/tmp/data/alice_images.pth', src=ALICE)
    x_bob_enc = crypten.load_from_party('/tmp/data/bob_images.pth', src=BOB)

    crypten.print(x_alice_enc.size())
    crypten.print(x_bob_enc.size())

    # Combine the feature sets: identical to Tutorial 3
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)

    # Reshape to match the network architecture
    x_combined_enc = x_combined_enc.unsqueeze(1)

    size = x_combined_enc.size()
    print("=====combined_size=======", size)
    # training hyperparams
    learning_rate = 0.0001
    num_epochs = 10
    batch_size = 64
    num_batches = x_combined_enc.size(0) // batch_size

    print_every = batch_size

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size, 1, 1))
    fixed_z = torch.from_numpy(fixed_z).float()
    fixed_z = crypten.cryptensor(fixed_z)

    input_size = (10, 100, 1, 1)
    x = crypten.cryptensor(torch.randn(input_size))
    G = Generator()
    G.encrypt()
    outputG = G(x)
    # print(f"Generated data shape: {outputG.shape}\n")

    input_size = (10, 1, 28, 28)
    x = crypten.cryptensor(torch.randn(input_size))
    D = Discriminator(nc=1, ndf=32)
    D.encrypt()
    outputD = D(x)
    # print(f"Discriminator output shape: {outputD.shape}")
    D.train() # Change to training mode
    G.train()

    rank = comm.get().get_rank()
    for i in range(num_epochs):
        crypten.print(f"Epoch {i} in progress:")

        for batch in range(num_batches):
            # define the start and end of the training mini_batch
            start, end = batch * batch_size, (batch + 1) * batch_size

            # construct CrypTensor out of training examples / labels
            real_images = x_combined_enc[start:end] * 2 - 1 # rescale input
            batch_size = real_images.size(0)

            # Monitor communication stats
            crypten.reset_communication_stats()
            comm_start_rounds = comm.get().get_communication_stats()['rounds']
            comm_start_bytes = comm.get().get_communication_stats()['bytes']

            # ===========================================
            #            TRAIN THR DISCRIMINATOR
            # ===========================================
            # perform forward pass:
            D.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            # smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size, 1, 1)) # size大小的矩阵
            z = torch.from_numpy(z).float()
            z = crypten.cryptensor(z)

            fake_images = G(z)

            # Compute the discriminator losses on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            rank = comm.get().get_rank()
            # crypten.print(f"\nRank{rank}:\n d_loss_dec:{d_loss.get_plain_text()}", in_order=True)
            d_loss.backward()
            D.update_parameters(learning_rate)
            # ========================================
            #            TRAIN THE GENERATOR
            # ========================================
            G.zero_grad()
            # Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size, 1, 1))
            z = torch.from_numpy(z).float()
            z = crypten.cryptensor(z)
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)

            # perform backprop
            g_loss.backward()
            G.update_parameters(learning_rate)

            # Print some loss stats
            if start % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        i+1, num_epochs, d_loss.get_plain_text().item(), g_loss.get_plain_text().item()))
                
            batch_d_loss = d_loss.get_plain_text()
            batch_g_loss = g_loss.get_plain_text()
            # crypten.print(f"\tBatch {(batch + 1)} of {num_batches} D_Loss {batch_d_loss.item():.4f}")
            # crypten.print(f"\tBatch {(batch + 1)} of {num_batches} G_Loss {batch_g_loss.item():.4f}")
            
            # After GAN training steps, gather communication stats
            comm_end_rounds = comm.get().get_communication_stats()['rounds']
            comm_end_bytes = comm.get().get_communication_stats()['bytes']
            # print("comm_end_bytes:", comm_end_bytes)
            # print(f"Epoch [{i}/{num_epochs}]")


        ## After each epoch ## 
        # append discriminator loss and generator loss
        losses.append((d_loss.get_plain_text().item(), g_loss.get_plain_text().item()))

        # generate and save sample, fake images
        G.eval()
        samples_z = G(fixed_z)
        samples.append(samples_z.get_plain_text())
        G.train() # back to train model

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    return losses

run_lpgan()