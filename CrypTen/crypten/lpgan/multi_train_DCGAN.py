import numpy as np
import torch
import crypten
import crypten.nn as nn
import crypten.mpc as mpc
import crypten.communicator as comm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, datasets, transforms
import pickle as pkl
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import time
import matplotlib.animation as animation
from IPython.display import HTML
from crypten.lpgan.Secfunctions import print_communication_costs, print_execution_times, timeit, inference_mode, model_scope
import os
from torch.utils.data import DataLoader

os.environ["GLOO_SOCKET_IFNAME"] = 'enp3s0'
os.environ["GLOO_IPV6"] = '0'

torch.cuda.empty_cache()

crypten.init()
# torch.set_num_threads(1)
torch.set_num_threads(8)  # 根据 CPU 核心数调整
# Alice is party 0
ALICE = 0
BOB = 1

dataroot = "../data/MNSIT"  # 数据集所在的路径，我们已经事先下载下来了
# dataroot = "../data/FashionMNIST"  # 数据集所在的路径，我们已经事先下载下来了
# dataroot = "../data/USPS"  # 数据集所在的路径，我们已经事先下载下来了
workers = 10  # 数据加载时的进程数
batch_size = 16  # 生成器输入的大小

image_size = 64  # 训练图像的大小
nc = 1  # 训练图像的通道数，彩色图像的话就是 3
nz = 100  # 输入是100 维的随机噪声 z，看作是 100 个 channel，每个特征图宽高是 1*1
ngf = 64  # 生成器中特征图的大小，
ndf = 64  # 判别器中特征图的大小
num_epochs = 1  # 训练的轮次
lr = 0.0005  # 学习率大小
beta1 = 0.5  # Adam 优化器的参数
ngpu = 1  # 可用 GPU  的个数，0 代表使用 CPU
device = torch.device('cpu')

class CustomMNISTDataset(Dataset):
    """封装(images, labels)"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def preprocess_data():
    """预处理数据集，并封装为trainloader"""
    train_data = datasets.MNIST(
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]),
        download=True  # 从互联网下载，如果已经下载的话，就直接使用
    )

    num_samples = len(train_data)
    print(f"Number of samples in the training dataset: {num_samples}")

    #Transform labels into one-hot encoding
    images, labels = take_samples(train_data, n_samples=2000)
    
    # 将它们封装为自定义的 Dataset 对象
    train_set = CustomMNISTDataset(images, labels)

    # 便于迭代数据 During each epoch, the model will go through all 1250 batches, effectively using all 5000 images.
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    testloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 看是否存在可用的 GPU
    device = torch.device('cpu')

            
    # 获取第一个样本
    sample_index = 0
    image, label = train_data[sample_index]

    # 转换为 NumPy 数组
    image_np = image.numpy()

    # # 显示图像
    # plt.imshow(image_np.squeeze(), cmap='gray')
    # plt.title(f"Label: {label}")
    # plt.axis('off')
    # plt.show()

    return trainloader, testloader

# 权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight.data) # new
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def encrypt_data_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        # S1
        input_upd = input[ :1000]
        # crypten.save_from_party(input_upd, "../data/bob_images.pth", src=BOB)  
        # private_input = crypten.load_from_party("../data/bob_images.pth", src=BOB)

    else:
        # S0
        input_upd = input[1000:2000]
        # crypten.save_from_party(input_upd, "../data/alice_images.pth", src=ALICE)
        # private_input = crypten.load_from_party("../data/alice_images.pth", src=BOB)

    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def take_samples(digits, n_samples):
    """Returns images and one-hot labels based on sample size"""
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
 
# 预先加密数据并保存
def preprocess_and_encrypt_data(trainloader):
    encrypted_data = []
    for input, target in trainloader:
        input = input.unsqueeze(1)  # 扩展维度
        real_images = input * 2 - 1  # 归一化到 [-1, 1)
        real_images = encrypt_data_tensor(real_images)  # 加密数据
        encrypted_data.append(real_images)
    
    # 保存加密数据
    with open('experiment/encrypted_real_images.pkl', 'wb') as f:
        pkl.dump(encrypted_data, f)


# class Discriminator(crypten.nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.nc = nc 
#         self.ndf = ndf
#         self.conv1 = crypten.nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)  # (64, 14, 14)
#         self.leaky_relu = crypten.nn.LeakyRelu(0.2)
#         self.conv2 = crypten.nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)  # (128, 7, 7)
#         self.bn1 = crypten.nn.BatchNorm2d(ndf * 2)
#         self.conv3 = crypten.nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)  # (256, 3, 3)
#         self.bn2 = crypten.nn.BatchNorm2d(ndf * 4)
#         self.conv4 = crypten.nn.Conv2d(ndf * 4, 1, 1, stride=1, padding=0, bias=False)  # (1, 1, 1) 28*28
#         self.sd = crypten.nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         # print("conv1(x):", x.get_plain_text())
#         x = self.leaky_relu(x)
#         # print("leaky_relu:", x.get_plain_text())
#         x = self.conv2(x)
#         # print("conv2(x):", x.get_plain_text())
#         x = self.bn1(x)
#         # print("bn1(x):", x.get_plain_text())
#         x = self.leaky_relu(x)
#         # print("leaky_relu:", x.get_plain_text())
#         x = self.bn2(self.conv3(x))
#         # print("bn2:", x.get_plain_text())
#         x = self.leaky_relu(x)
#         # print("leaky_relu:", x.get_plain_text())
#         x = self.sd(self.conv4(x))
#         # print("sd:", x.get_plain_text())

#         return x.view(x.size(0), -1).mean(dim=1)


# class Generator(crypten.nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.decon1 = crypten.nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0,bias=False)
#         self.bn1 = crypten.nn.BatchNorm2d(512)
#         self.decon2 = crypten.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)  
#         self.bn2 = crypten.nn.BatchNorm2d(256)
#         self.decon3 = crypten.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)  
#         self.bn3 = crypten.nn.BatchNorm2d(128)
#         self.decon4 = crypten.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)  
#         self.bn4 = crypten.nn.BatchNorm2d(64)
#         self.decon5 = crypten.nn.ConvTranspose2d(64, nc, 4, stride=2, padding=1, bias=False)
#         self.relu = crypten.nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.decon1(x))
#         x = self.bn1(x)
#         # print(f"Shape after decon1: {x.shape}")
#         x = self.relu(self.decon2(x))
#         x = self.bn2(x)
#         # print(f"Shape after deconv2: {x.shape}")
#         x = self.relu(self.decon3(x))
#         x = self.bn3(x)
#         # print(f"Shape after deconv3: {x.shape}")
#         x = self.relu(self.decon4(x))
#         # print("After relu:\n", x.get_plain_text())
#         x = self.bn4(x)
#         x = self.decon5(x)
#         # print(f"Shape after deconv5: {x.shape}")
#         x = x.tanh()
#         return x


class Generator(crypten.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = crypten.nn.Sequential(
            crypten.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            crypten.nn.BatchNorm2d(ngf * 8),
            crypten.nn.ReLU(),
            crypten.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            crypten.nn.BatchNorm2d(ngf * 4),
            crypten.nn.ReLU(),
            crypten.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            crypten.nn.BatchNorm2d(ngf * 2),
            crypten.nn.ReLU(),
            crypten.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            crypten.nn.BatchNorm2d(ngf),
            crypten.nn.ReLU(),
            # crypten.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # CelebA
            crypten.nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=False),  # MNIST
            
        )

    def forward(self, x):
        res = self.main(x)
        return res.tanh()

class Discriminator(crypten.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = crypten.nn.Sequential(

            crypten.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            crypten.nn.LeakyRelu(0.2),
            
            crypten.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            crypten.nn.BatchNorm2d(ndf * 2),
            crypten.nn.LeakyRelu(0.2),
     
            crypten.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            crypten.nn.BatchNorm2d(ndf * 4),
            crypten.nn.LeakyRelu(0.2),
 
            # crypten.nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),  # Celeba cira10
            crypten.nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),  # MNIST

            crypten.nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)  # Output shape: [128, 1, 5, 5]
        # output = output.view(output.size(0), -1).mean(dim=1)  # Flatten and take mean
        output = output.view(-1,1).squeeze(1)  # Flatten and take mean
        return output


# class Discriminator(crypten.nn.Module):

#     def __init__(self):

#         super(Discriminator, self).__init__()


#         # self.nc(1 or 3) * 64 * 64
#         self.conv1 = crypten.nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)
#         self.leaky_relu = crypten.nn.LeakyRelu(0.2)
#         # self.ndf(64) * 32 * 32
#         self.conv2 = crypten.nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)
#         self.bn1 = crypten.nn.BatchNorm2d(ndf * 2)

#         # self.ndf*2(128) * 16 * 16
#         self.conv3 = crypten.nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)
#         self.bn2 = crypten.nn.BatchNorm2d(ndf * 4)
#         # self.ndf*4(256) * 8 * 8
#         self.conv4 = crypten.nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)
#         self.bn3 = crypten.nn.BatchNorm2d(ndf * 8)
#         # self.ndf*8(512) * 4 * 4, stride = 1
#         # self.conv5 = crypten.nn.Conv2d(self.ndf * 8, 1, 4, stride=1, padding=0, bias=False) # 如果输入是64x64
#         self.conv5 = crypten.nn.Conv2d(ndf * 8, 1, 1, stride=1, padding=0, bias=False) # 如果输入是28x28
#         self.sd = crypten.nn.Sigmoid()
#         # self.dp = crypten.nn.Dropout(0.3)

#     def forward(self, x):
        
#         x = self.leaky_relu(self.conv1(x))
#         # print(f"Shape after conv1: {x.shape}")
#         x = self.leaky_relu(self.bn1(self.conv2(x)))
#         # print(f"Shape after conv2: {x.shape}")
#         x = self.leaky_relu(self.bn2(self.conv3(x)))
#         # print(f"Shape after conv3: {x.shape}")
#         x = self.leaky_relu(self.bn3(self.conv4(x)))
#         # print(f"Shape after conv4: {x.shape}")
#         x = self.sd(self.conv5(x))
#         # x = self.dp(self.conv5(x))
        
#         # print(f"Shape after conv5: {x.shape}")

#         return x.view(-1, 1).squeeze(1)




def construct_private_model(modelD, modelG):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        D = modelD
        # D.load_state_dict(torch.load('model/modelD2.pt',map_location=torch.device('cpu')))
        D.load_state_dict(torch.load('model/netD_epoch_14.pth',map_location=torch.device('cpu')))
        # D.load_state_dict(torch.load('model/netD_usps.pth',map_location=torch.device('cpu')))
        # D.load_state_dict(torch.load('model/netD_fashion.pth',map_location=torch.device('cpu')))
        # D.load_state_dict(torch.load('model/modelD_cifar.pt',map_location=torch.device('cpu')))
        # G = modelG
        G = Generator()
    else:
        # 占位用的，就和份额的broadcast一个道理
        # need to provide a dummy model to tell CrypTen the model's structure
        D = Discriminator()
        # G = Generator()
        G = modelG
        # G.load_state_dict(torch.load('model/modelG2.pt',map_location=torch.device('cpu')))
        G.load_state_dict(torch.load('model/netG_epoch_14.pth',map_location=torch.device('cpu')))
        # G.load_state_dict(torch.load('model/netG_usps.pth',map_location=torch.device('cpu')))
        # G.load_state_dict(torch.load('model/netG_fashion.pth',map_location=torch.device('cpu')))
        # G.load_state_dict(torch.load('model/modelG_cifar.pt',map_location=torch.device('cpu')))
        
    private_modelD = D.encrypt(src=0)
    # private_modelG = G.encrypt(src=0)
    private_modelG = G.encrypt(src=1) # 如果一方是G，一方是D会不会内存会小一点？

    return private_modelD, private_modelG


from torchvision import models, transforms
import torch.nn.functional as F
import torch

# 加载预训练的Inception模型
inception_model = models.inception_v3(weights='DEFAULT', transform_input=False)  # 使用 weights 代替 pretrained
inception_model.eval()

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义Inception Score计算函数
def inception_score(img_list, inception_model, num_splits=10):
    inception_model.eval()
    preds = []

    with torch.no_grad():
        for img in img_list:
            # 如果图像是单通道，扩展为三个通道
            if img.size(0) == 1:  # 检查是否为单通道图像
                img = img.expand(3, -1, -1)  # 复制单通道数据到三个通道
            
            img = transform(img)
            img = img.unsqueeze(0)  # 增加batch维度
            img = img.to('cpu')  # 如果inception模型在cpu上
            pred = inception_model(img)[0]
            preds.append(pred)

    preds = torch.stack(preds)
    preds = F.softmax(preds, dim=1)
    split_scores = []
    for k in range(num_splits):
        part = preds[k * (len(preds) // num_splits): (k + 1) * (len(preds) // num_splits), :]
        p_y = part.mean(0)
        kl_divergence = part * (torch.log(part) - torch.log(p_y.unsqueeze(0)))
        kl_divergence = kl_divergence.sum(1)
        split_scores.append(kl_divergence.mean().exp())

    return torch.tensor(split_scores).mean().item(), torch.tensor(split_scores).std().item()


from scipy import linalg

# 定义特征提取函数
def get_features(images, model, device):
    # 调整图像大小为 299x299
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # 如果图像值在 [-1, 1]，调整到 [0, 1]（根据你的模型输出调整）
    if images.min() < 0:
        images = (images + 1) / 2  # 从 [-1, 1] 转为 [0, 1]
    # 归一化到 Inception v3 的输入范围（通常需要进一步标准化，但这里简化）
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(real_features, generated_features):
    # 计算均值
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    # 计算协方差
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # 计算FID
    diff = mu_real - mu_generated
    covmean = linalg.sqrtm(sigma_real.dot(sigma_generated), disp=False)[0]
    if np.iscomplexobj(covmean):  # 处理复数情况
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid


# Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
        
    # numerically stable loss
    criterion = crypten.nn.ESBCELoss()
    # criterion = crypten.nn.BCELoss()
    # criterion = crypten.nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    # criterion = crypten.nn.BCEWithLogitsLoss()
    # criterion = crypten.nn.BCELoss()
    criterion = crypten.nn.ESBCELoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss



def clip_gradient(grad, max_norm=10.0):
    """
    Clipping gradients to avoid exploding gradients during training.
    """
    grad = grad.get_plain_text()
    grad_norm = grad.norm(2)  # L2 norm of the gradient
    if grad_norm > max_norm:
        grad = grad * (max_norm / grad_norm)  # Scale the gradients
    grad = crypten.cryptensor(grad)
    # print("==========clip===========")
    return grad

def normalize_gradient(grad, norm_type=2):
    """
    Normalize gradients to avoid vanishing/exploding gradients.
    """
    grad = grad.get_plain_text()
    if norm_type == 2:
        grad_norm = grad.norm(2)  # L2 norm
    else:
        raise NotImplementedError(f"Norm type {norm_type} is not implemented")

    if grad_norm > 10.0:
        grad = grad / grad_norm  # Normalize the gradient to unit norm
    grad = crypten.cryptensor(grad)
    # print("==========normalize===========")

    return grad



import os
# @mpc.run_multiprocess(world_size=2)
@timeit
def run_lpgan():
    crypten.init()
    os.environ['GLOO_TIMEOUT'] = '500'

    G = Generator().to(device)
    # G.apply(weights_init)
    for layer in G.children():
        weights_init(layer)

    D = Discriminator().to(device)
    # D.apply(weights_init)
    for layer in D.children():
        weights_init(layer)
    D, G = construct_private_model(D, G) 

    criterion = crypten.nn.BCELoss()  # 定义损失函数

    # 创建一批潜在向量，来可视化生成器的生成过程
    fixed_noise = crypten.cryptensor(torch.randn(100, nz, 1, 1, device=device))

    real_label = 1.  # “真”标签
    fake_label = 0.  # “假”标签

    # # 4ceng
    # # Inception Score: 1.3116 ± 0.0346
    # optimizerG = crypten.optim.SGD(G.parameters(), lr=1e-4, momentum=0.9)
    # optimizerD = crypten.optim.SGD(D.parameters(), lr=2e-4, momentum=0.8)

    # # baozha
    # optimizerG = crypten.optim.SGD(G.parameters(), lr=2e-4)
    # optimizerD = crypten.optim.SGD(D.parameters(), lr=2e-4)

    # # 5ceng 1.4776 ± 0.0318
    # optimizerG = crypten.optim.SGD(G.parameters(), lr=2e-4, momentum=0.8)
    # optimizerD = crypten.optim.SGD(D.parameters(), lr=3e-4, momentum=0.9)

    # # 20000 Inception Score: 1.4723 ± 0.0038 
    # optimizerG = crypten.optim.SGD(G.parameters(), lr=2e-4, momentum=0.8)
    # optimizerD = crypten.optim.SGD(D.parameters(), lr=8e-4, momentum=0.9)

    # # 50000, explod the range of gradient's clip and normalize
    # optimizerG = crypten.optim.SGD(G.parameters(), lr=2e-4, momentum=0.8)
    # optimizerD = crypten.optim.SGD(D.parameters(), lr=1e-3, momentum=0.9)

    G_lr = 2e-2
    D_lr = 2e-2
    scaling_factor = 5 # 1PC change here #
    r_lr = 1.1
    threshold = 2.5 # 1PC change here #

    # 50000
    optimizerG = crypten.optim.SGD(G.parameters(), lr=G_lr)
    optimizerD = crypten.optim.SGD(D.parameters(), lr=D_lr)
    
    # 定义一些变量，用来存储每轮的相关值
    img = []
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep = 10
    total_time = 0
    print("Starting Training Loop...")

    # 加载加密数据
    rank = comm.get().get_rank()
    if rank == ALICE:
        # with open('experiment/alice_5000_FashionMNIST.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/alice_5000_USPS.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/alice_b64_5000.pkl', 'rb') as f: # 1PC change here #
        with open('experiment/alice_b64_100.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/alice_100_cifar.pkl', 'rb') as f: # 1PC change here #
            encrypted_real_images = pkl.load(f)
    elif rank == BOB:
        # with open('experiment/bob_5000_FashionMNIST.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/bob_5000_USPS.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/bob_b64_5000.pkl', 'rb') as f: # 1PC change here #
        with open('experiment/bob_b64_100.pkl', 'rb') as f: # 1PC change here #
        # with open('experiment/bob_100_cifar.pkl', 'rb') as f: # 1PC change here #
            encrypted_real_images = pkl.load(f)

    # D.train() # Change to training mode
    # G.train()


    # net_io_before = get_network_bytes()
    # for epoch in range(num_epochs):
    #     start_time = time.time()  # 记录当前 epoch 开始的时间
    #     for i, real_images in enumerate(encrypted_real_images):
    #         batch_size = real_images.size(0)
    #         # ===========================================
    #         #            TRAIN THR DISCRIMINATOR
    #         # ===========================================
    #         # perform forward pass:
    #         D.zero_grad()
    #         # 1. Train with real images

    #         # Compute the discriminator losses on real images
    #         start_time1 = time.time()
    #         # print("real_images'size:", real_images.size())
    #         D_real = D(real_images.to(device)).view(-1)
    #         end_time = time.time()
    #         runtime = end_time - start_time1
    #         # print(f"Discriminator executed in {runtime:.4f} seconds")
            
    #         d_real_loss = real_loss(D_real, smooth=True)# smooth the real labels

    #         d_real_loss.backward()

    #         # 2. Train with fake images

    #         # # Generate fake images
    #         z = torch.randn(batch_size, nz, 1, 1, device=device)
            
    #         print("batch_size:", batch_size)
    #         z = crypten.cryptensor(z)
    #         torch.set_printoptions(precision=16)
            
    #         start_time2 = time.time()
    #         fake_images = G(z)
    #         end_time = time.time()
    #         runtime = end_time - start_time2
    #         print(f"Generator executed in {runtime:.4f} seconds")
            
    #         # print("G_z:", fake_images.get_plain_text())
    #         # Compute the discriminator losses on fake images
    #         D_fake = D(fake_images.detach()).view(-1)
    #         d_fake_loss = fake_loss(D_fake)
    #         # print("d_fake:", D_fake.get_plain_text())
    #         # print("d_fake_loss:", d_fake_loss.get_plain_text())

    #         d_fake_loss.backward()

    #         # Clip gradients for discriminator
    #         for param in D.parameters():
    #             if param.grad is not None:
    #                 param.grad = clip_gradient(param.grad)
    #         # Normalize gradients for discriminator
    #         for param in D.parameters():
    #             if param.grad is not None:
    #                 param.grad = normalize_gradient(param.grad)

    #         # add up loss and perform backprop
    #         d_loss = d_real_loss + d_fake_loss
    #         # d_loss.backward()


    #         optimizerD.step()
    #         # D.update_parameters(2e-4)
    #         # ========================================
    #         #            TRAIN THE GENERATOR
    #         # ========================================
    #         G.zero_grad()
    #         # Train with fake images and flipped labels
    #         # Compute the discriminator losses on fake images
    #         # using flipped labels!
    #         D_fake = D(fake_images).view(-1)
    #         g_loss = real_loss(D_fake)
    #         # print("D_fake:", D_fake.get_plain_text())
    #         # print("g_Loss:", g_loss.get_plain_text())

    #         # perform backprop
    #         g_loss.backward()

    #         # Clip gradients for generator
    #         for param in G.parameters():
    #             if param.grad is not None:
    #                 param.grad = clip_gradient(param.grad)
    #         # Normalize gradients for generator
    #         for param in G.parameters():
    #             if param.grad is not None:
    #                 param.grad = normalize_gradient(param.grad)

    #         optimizerG.step()
    #         # G.update_parameters(2e-4)

    #         # Print some loss stats
    #         end_time = time.time()
    #         run_time = round(end_time - start_time)
    #         print(f'Epoch:[{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
    #               f'Step:[{i+1:0>{len(str(len(encrypted_real_images)))}}/{len(encrypted_real_images)}]',
    #               f'Loss-D:{d_loss.get_plain_text().item():.4f}',
    #               f'Loss-G:{g_loss.get_plain_text().item():.4f}',
    #             #   f'D(x):{D_x:.4f}',
    #             #   f'D(G(z)):[{D_G_z1:.4f}/{D_G_z2:.4f}]',
    #               f'Time:{run_time}s',
    #               end='\r')
            
    #         G_losses.append(g_loss.get_plain_text().item())
    #         D_losses.append(d_loss.get_plain_text().item())

    #         # D_x_list.append(D_x)
    #         # D_z_list.append(D_G_z2)
    #         torch.save(G.state_dict(), 'model/fpgan_g.pt')
    #         torch.save(D.state_dict(), 'model/fpgan_d.pt')

    #         # 保存最好的模型
    #         if g_loss.get_plain_text() < loss_tep:
    #             torch.save(G.state_dict(), 'model/lpgan5000.pt')
    #             temp = g_loss
    #         total_time += run_time
    #     with torch.no_grad():
    #         fake = G(fixed_noise).detach()
    #         fake = fake.get_plain_text()
    #     img_list.append(utils.make_grid(fake*0.5+0.5, nrow=10))
    #     img.extend(fake)  # 保存每张生成图像
    #     print()

    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # print("traffic_training:",traffic_diff)
    # print("runtime_training:",total_time)

    # # 训练结束后进行测试
    # test_batch_sizes = [1, 16, 32, 64]  # 需要测试的batchsize列表
    # test_batch_sizes = [1]  # 需要测试的batchsize列表
    # test_batch_sizes = [16]  # 需要测试的batchsize列表
    # test_batch_sizes = [32]  # 需要测试的batchsize列表
    test_batch_sizes = [64]  # 需要测试的batchsize列表
    num_test_runs = 1  # 每个batchsize的运行次数

    print("\nTesting Generator inference time...")
    G = Generator()
    G.encrypt()
    G.load_state_dict(torch.load('model/fpgan_g.pt', map_location = torch.device('cpu')))
    G.eval()  # 设置为评估模式
    for bs in test_batch_sizes:
        total_time = 0.0
        for _ in range(num_test_runs):
            # 生成加密噪声输入
            z = torch.randn(bs, nz, 1, 1)
            z_enc = crypten.cryptensor(z)
            # 计时前向传播
            start_time = time.time()
            with crypten.no_grad():
                with model_scope("Generator"):
                    with inference_mode():
                        _ = G(z_enc)
            end_time = time.time()
            total_time += end_time - start_time
        avg_time = total_time / num_test_runs
        print(f"Batch Size {bs}: Generator Average Time {avg_time:.4f} seconds")

    print("\nTesting Discriminator inference time...")
    D = Discriminator()
    D.encrypt()
    D.load_state_dict(torch.load('model/fpgan_d.pt', map_location = torch.device('cpu')))
    D.eval()  # 设置为评估模式
    for bs in test_batch_sizes:
        total_time = 0.0
        for _ in range(num_test_runs):
            # 生成加密图像输入（假设图像尺寸为1x64x64）
            images = torch.randn(bs, nc, 64, 64)
            images_enc = crypten.cryptensor(images)
            # 计时前向传播
            start_time = time.time()
            with crypten.no_grad():
                with model_scope("Doscriminator"):
                    with inference_mode():
                        _ = D(images_enc)
            end_time = time.time()
            total_time += end_time - start_time
        avg_time = total_time / num_test_runs
        print(f"Batch Size {bs}: Discriminator Average Time {avg_time:.4f} seconds")

    
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses[::100], label="G")
    # plt.plot(D_losses[::100], label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.axhline(y=0, label="0", c="g")  # asymptote
    # plt.legend()
    # plt.savefig('../lpgan/result/loss_100.jpg')

    # fig = plt.figure(figsize=(10, 10))
    # plt.axis("off")
    # ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
    # ani = animation.ArtistAnimation(fig,
    #                                 ims,
    #                                 interval=1000,
    #                                 repeat_delay=1000,
    #                                 blit=True)
    # HTML(ani.to_jshtml())


    # # Save the comparation result
    # # plt.savefig('result/fake_USPS_CrypTen_5000.jpg', bbox_inches='tight')
    # # 在训练循环结束后评估Inception Score
    # mean_inception_score, std_inception_score = inception_score(img, inception_model)
    # print(f"Inception Score: {mean_inception_score:.4f} ± {std_inception_score:.4f}")

    # # 在训练循环结束后添加 FID 评估
    # print("Evaluating FID...")
    # num_images = 100

    # # #loading the dataset
    # # dataset = datasets.MNIST(root='../data/MNIST', download=True,
    # #                     transform=transforms.Compose([
    # #                         transforms.Resize(28),
    # #                         transforms.ToTensor(),
    # #                         transforms.Normalize((0.5,), (0.5,)),
    # #                     ]))

    # # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
    # #                                         shuffle=True, num_workers=2)

    # # 训练集加载并进行归一化等操作
    # dataset = datasets.USPS(root=dataroot, download=True,
    #                     transform=transforms.Compose([
    #                         transforms.Resize(28),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5,), (0.5,)),
    #                     ]))

    # # 数据加载器，训练过程中不断产生数据
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         num_workers=workers)

    # # 准备真实图像
    # real_images = []
    # for i, data in enumerate(dataloader):
    #     real_images.append(data[0])
    #     if len(real_images) * dataloader.batch_size >= num_images:
    #         break
    # real_images = torch.cat(real_images, dim=0)[:num_images]

    # # 准备生成图像
    # noise = torch.randn(num_images, nz, 1, 1,device=device)
    # noise = crypten.cryptensor(noise)

    # G.to(device) 
    # with torch.no_grad():
    #     generated_images = G(noise).detach()
    # # # 加载 Inception v3 模型
    # # inception_model = models.inception_v3(pretrained=True).to(device)
    # # inception_model.eval()

    # # print("real_images shape:", real_images.shape)
    # # print("generated_images shape:", generated_images.shape)
    # if real_images.shape[1] == 1:
    #     real_images = real_images.repeat(1, 3, 1, 1)  # 将 1 通道重复 3 次
    # if generated_images.shape[1] == 1:
    #     generated_images = generated_images.repeat(1, 3, 1, 1)
    # generated_images = generated_images.get_plain_text()
    # # 提取特征
    # real_features = get_features(real_images, inception_model, device)
    # generated_features = get_features(generated_images, inception_model, device)

    # # 计算 FID
    # fid_score = calculate_fid(real_features, generated_features)
    # print(f'FID Score: {fid_score:.4f}')

    print_communication_costs()
    print_execution_times()
run_lpgan()