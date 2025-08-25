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


crypten.init()
torch.set_num_threads(1)
# Alice is party 0
ALICE = 0
BOB = 1

dataroot = "../../data"  # 数据集所在的路径，我们已经事先下载下来了
workers = 10  # 数据加载时的进程数
batch_size = 64  # 生成器输入的大小

image_size = 64  # 训练图像的大小
nc = 1  # 训练图像的通道数，彩色图像的话就是 3
nz = 100  # 输入是100 维的随机噪声 z，看作是 100 个 channel，每个特征图宽高是 1*1
ngf = 64  # 生成器中特征图的大小，
ndf = 32  # 判别器中特征图的大小
num_epochs = 1  # 训练的轮次
lr = 0.0005  # 学习率大小
beta1 = 0.5  # Adam 优化器的参数
ngpu = 1  # 可用 GPU  的个数，0 代表使用 CPU

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
    images, labels = take_samples(train_data, n_samples=100)
    
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
    device = torch.device('cuda:0' if (
        torch.cuda.is_available() and ngpu > 0) else 'cpu')

            
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
        input_upd = input[ :50]
        # crypten.save_from_party(input_upd, "data/bob_images.pth", src=BOB)  
        # private_input = crypten.load_from_party("../data/bob_images.pth", src=BOB)

    else:
        # S0
        input_upd = input[50:100]
        # crypten.save_from_party(input_upd, "data/alice_images.pth", src=ALICE)
        # private_input = crypten.load_from_party("../data/alice_images.pth", src=BOB)

    # p1 = crypten.load_from_party("data/bob_images.pth", src=BOB)
    # p0 = crypten.load_from_party("data/alice_images.pth", src=ALICE)
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input
    # return p0,p1


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
    """Preprocess and encrypt data for each party separately."""
    rank = comm.get().get_rank()
    
    # 每个参与方各自的加密数据列表
    encrypted_data = []

    for input, target in trainloader:
        input = input.unsqueeze(1)  # 扩展维度
        real_images = input
        encrypted_images = encrypt_data_tensor(real_images)  # 加密数据
        # p0,p1 = encrypt_data_tensor(real_images)  # 加密数据
        encrypted_data.append(encrypted_images)  # 追加到列表中

    # 根据参与方的身份保存加密数据
    if rank == ALICE:
        '''
            上述做法，应该是将src=0和src=1的数据分别保存了，
            但这并不是data[:500]和data[500:]分别保存，
            而是data的src=0和src=1的分别保存
            
            但是CrypTensor不能直接使用 pickle 保存和加载
            所以还是只能以份额的形式保存（也就是最开始的做法）
        '''
        with open('experiment/encrypted_real_images_alice_100.pkl', 'wb') as f:
            pkl.dump(encrypted_data, f)
    elif rank == BOB:
        with open('experiment/encrypted_real_images_bob_100.pkl', 'wb') as f:
            pkl.dump(encrypted_data, f)


# @mpc.run_multiprocess(world_size=2)
def run():
    # 预处理数据集并在训练前加密
    trainloader, testloader = preprocess_data()
    preprocess_and_encrypt_data(trainloader)
    # bob_img = crypten.load_from_party('data/bob_images.pth',src=BOB)
    # with open('experiment/encrypted_real_images_bob.pkl', 'wb') as f:
    #     pkl.dump(bob_img, f)       
    # with open('experiment/encrypted_real_images_bob.pkl', 'rb') as f:
    #     bob = pkl.load(f)
    # # 显示加密后的 bob_img
    # encrypted_image = bob[0,0] # 第一个批次的第一个图像
    # decrypted_image = encrypted_image.get_plain_text()
    # # 显示图像
    # plt.imshow(decrypted_image.numpy())
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()


run()