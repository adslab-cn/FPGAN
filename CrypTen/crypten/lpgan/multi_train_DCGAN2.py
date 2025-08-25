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
from crypten.lpgan.Secfunctions import get_network_bytes, calculate_network_traffic, print_communication_costs
import torch

# print(torch.cuda.is_available())  # 检查 CUDA 是否可用
# print(torch.cuda.device_count())  # 检查 GPU 设备数量

    
# crypten.init()
# torch.set_num_threads(1)
# Alice is party 0
ALICE = 0
BOB = 1

dataroot = "../data"  # 数据集所在的路径，我们已经事先下载下来了
workers = 10  # 数据加载时的进程数
batch_size = 64  # 生成器输入的大小

image_size = 64  # 训练图像的大小
nc = 1  # 训练图像的通道数，彩色图像的话就是 3
nz = 100  # 输入是100 维的随机噪声 z，看作是 100 个 channel，每个特征图宽高是 1*1
ngf = 64  # 生成器中特征图的大小，
ndf = 64  # 判别器中特征图的大小
num_epochs = 1  # 训练的轮次
lr = 0.0005  # 学习率大小
beta1 = 0.5  # Adam 优化器的参数
ngpu = 0  # 可用 GPU  的个数，0 代表使用 CPU

# 看是否存在可用的 GPU
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
    images, labels = take_samples(train_data, n_samples=10000)
    
    # 将它们封装为自定义的 Dataset 对象
    train_set = CustomMNISTDataset(images, labels)

    # 便于迭代数据 During each epoch, the model will go through all 1250 batches, effectively using all 5000 images.
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    testloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

            
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

    return trainloader, testloader

# 权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
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
        input_upd = input[ :500]
        # crypten.save_from_party(input_upd, "../data/bob_images.pth", src=BOB)  
        # private_input = crypten.load_from_party("../data/bob_images.pth", src=BOB)

    else:
        # S0
        input_upd = input[500:1000]
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
 


class Generator(crypten.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decon1 = crypten.nn.ConvTranspose2d(in_channels=nz, 
                                                 out_channels=ngf*8,
                                                 kernel_size=4, 
                                                 stride=1, 
                                                 padding=0,
                                                 bias=False)
        self.bn1 = crypten.nn.BatchNorm2d(ngf*8)
        self.decon2 = crypten.nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=1, bias=False)  
        self.bn2 = crypten.nn.BatchNorm2d(ngf*4)
        self.decon3 = crypten.nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=1, bias=False)  
        self.bn3 = crypten.nn.BatchNorm2d(ngf*2)
        self.decon4 = crypten.nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=1, bias=False)  
        self.bn4 = crypten.nn.BatchNorm2d(ngf)
        self.decon5 = crypten.nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1, bias=False)
        self.relu = crypten.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.decon1(x))
        x = self.bn1(x)
        x = self.relu(self.decon2(x))
        x = self.bn2(x)
        x = self.relu(self.decon3(x))
        x = self.bn3(x)
        x = self.relu(self.decon4(x))
        x = self.bn4(x)
        x = self.decon5(x)
        x = x.tanh()
        return x



class Discriminator(crypten.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nc = nc 
        self.ndf = ndf
        self.conv1 = crypten.nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)  # (64, 14, 14)
        self.leaky_relu = crypten.nn.LeakyRelu(0.2)
        self.conv2 = crypten.nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)  # (128, 7, 7)
        self.bn1 = crypten.nn.BatchNorm2d(ndf * 2)
        self.conv3 = crypten.nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)  # (256, 3, 3)
        self.bn2 = crypten.nn.BatchNorm2d(ndf * 4)
        # self.conv4 = crypten.nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=0, bias=False)  # (1, 1, 1)
        self.conv4 = crypten.nn.Conv2d(ndf * 4, 1, 1, stride=1, padding=0, bias=False)  # (1, 1, 1)
        self.sd = crypten.nn.Sigmoid()

    def forward(self, x):
        # print("=====x'size:", x.size())
        x = self.leaky_relu(self.conv1(x))
        # print("=====After leakyrelu x'size:", x.size())
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.sd(self.conv4(x))

        return x.view(x.size(0), -1).mean(dim=1)



def add_state_dict_hooks(module):
    """递归为模型中的所有子模块添加 `_state_dict_hooks` 属性。"""
    if not hasattr(module, "_state_dict_hooks"):
        module._state_dict_hooks = {}
    for child in module.children():
        add_state_dict_hooks(child)    
        

def construct_private_model(modelD, modelG):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(1, 1, 28, 28)
    
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        D = modelD
        D.load_state_dict(torch.load('modelD.pt'), strict=True)

        # G = modelG
        G = Generator()
        
        # G = Generator(ngpu).to(device)
        # add_state_dict_hooks(G)
    else:
        # 占位用的，就和份额的broadcast一个道理
        # need to provide a dummy model to tell CrypTen the model's structure
        D = Discriminator()
        # D = Discriminator(ngpu).to(device)
        # add_state_dict_hooks(D)
        # G = Generator()
        G = modelG
        G.load_state_dict(torch.load('modelG.pt'), strict=True)

        
    private_modelD = D.encrypt(src=0)
    private_modelG = G.encrypt(src=0)
    # private_modelG = G.encrypt(src=1) # 如果一方是G，一方是D会不会内存会小一点？
    
    # private_modelD = crypten.nn.from_pytorch(D, dummy_input)
    # private_modelG = crypten.nn.from_pytorch(G, dummy_input)
    # print("pr_D", private_modelD)
    return private_modelD.encrypt(), private_modelG.encrypt()


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
    criterion = crypten.nn.BCELoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = crypten.nn.BCELoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

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


def clip_gradient(grad, max_norm=10.0):
    """
    Clipping gradients to avoid exploding gradients during training.
    """
    grad = grad.get_plain_text() # 也许这里也可以有工作——》在FSS下的安全梯度裁剪
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

# @mpc.run_multiprocess(world_size=2)
def run_lpgan():
    
    crypten.init()
    print("++++++++++++")
    
    G = Generator().to(device)
    # G = Generator(ngpu).to(device)

    # add_state_dict_hooks(G)
    # G.apply(weights_init)
    for layer in G.children():
        weights_init(layer)

    D = Discriminator().to(device)
    # D = Discriminator(ngpu).to(device)
    # add_state_dict_hooks(D)
    # D.apply(weights_init)
    for layer in D.children():
        weights_init(layer)
    D, G = construct_private_model(D, G) 

    criterion = crypten.nn.BCELoss()  # 定义损失函数

    # 创建一批潜在向量，我们将使用它们来可视化生成器的生成过程
    fixed_noise = crypten.cryptensor(torch.randn(100, nz, 1, 1, device=device))

    real_label = 1.  # “真”标签
    fake_label = 0.  # “假”标签

    # 为生成器和判别器定义优化器
    optimizerG = crypten.optim.SGD(G.parameters(), lr=2e-2)
    optimizerD = crypten.optim.SGD(D.parameters(), lr=2e-2)
    
    # 定义一些变量，用来存储每轮的相关值
    img = []
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep = 10
    print("Starting Training Loop...")

    # 加载加密数据
    rank = comm.get().get_rank()
    if rank == ALICE:
        '''
        误打误撞还弄对了，因为无法将加密后[:500]单独存到文件中，这是因为：CrypTensor依赖于分布式环境和特定的加密上下文，因此不能直接使用 pickle 保存和加载
        所以我只能保存份额
        应该这么解释：不能在节点1上，还存有节点0的数据。也无法将两个节点的数据同时存在一个文件中。
        '''
        with open('experiment/encrypted_real_images_alice_100.pkl', 'rb') as f:
            encrypted_real_images = pkl.load(f)
    elif rank == BOB:
        with open('experiment/encrypted_real_images_bob_100.pkl', 'rb') as f:
            encrypted_real_images = pkl.load(f)

    D.train() # Change to training mode
    G.train()

    # encrypted_image = encrypted_real_images[0][0]  # 第一个批次的第一个图像
    # decrypted_image = encrypted_image.get_plain_text()

    # # 反归一化到 [0, 1] 范围
    # normalized_image = (decrypted_image + 1) / 2

    # # 转换为 HWC 格式
    # hwc_image = normalized_image.permute(1, 2, 0)  # 从 CHW 转换为 HWC

    # # 显示图像
    # plt.imshow(hwc_image.numpy())
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    
    net_io_before = get_network_bytes()
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录当前 epoch 开始的时间
        for i, real_images in enumerate(encrypted_real_images):
            batch_size = real_images.size(0)
            # ===========================================
            #            TRAIN THR DISCRIMINATOR
            # ===========================================
            # perform forward pass:

            # =================================
            #        输出解密后的图像
            # =================================
            # decrypted_image = real_images[0].get_plain_text()
            # # 反归一化到 [0, 1] 范围
            # normalized_image = (decrypted_image + 1) / 2
            # # 转换为 HWC 格式
            # hwc_image = normalized_image.permute(1, 2, 0)  # 从 CHW 转换为 HWC
            # # 显示图像
            # plt.imshow(hwc_image.numpy())
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            #===================================

            D.zero_grad()
            # 1. Train with real images

            # Compute the discriminator losses on real images
            # real_images = encrypt_data_tensor(real_images)
            net_io_before = get_network_bytes()
            D_real = D(real_images.to(device)).view(-1)
            net_io_after = get_network_bytes()
            traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
            print("=====traffic_D=====:",traffic_diff)
            
            # d_real_loss = criterion(D_real, label)# smooth the real labels
            d_real_loss = real_loss(D_real, smooth=True)# smooth the real labels
            d_real_loss.backward()

            # 2. Train with fake images

            # Generate fake images
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            z = crypten.cryptensor(z)

            fake_images = G(z)
            print("fakeG")

            # Compute the discriminator losses on fake images
            D_fake = D(fake_images.detach()).view(-1)
            d_fake_loss = fake_loss(D_fake)
            d_fake_loss.backward()

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            # d_loss.backward()

            # Clip gradients for discriminator
            for param in D.parameters():
                if param.grad is not None:
                    param.grad = clip_gradient(param.grad)
            # Normalize gradients for discriminator
            for param in D.parameters():
                if param.grad is not None:
                    param.grad = normalize_gradient(param.grad)

            optimizerD.step()
            # D.update_parameters(8e-3)
            # ========================================
            #            TRAIN THE GENERATOR
            # ========================================
            G.zero_grad()
            # Train with fake images and flipped labels
            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images).view(-1)
            g_loss = real_loss(D_fake)

            # perform backprop
            g_loss.backward()

            # Clip gradients for generator
            for param in G.parameters():
                if param.grad is not None:
                    param.grad = clip_gradient(param.grad)
            # Normalize gradients for generator
            for param in G.parameters():
                if param.grad is not None:
                    param.grad = normalize_gradient(param.grad)

            # D_G_z2 = D_fake.mean().get_plain_text.item()
            optimizerG.step()
            # G.update_parameters(2e-2)

            # Print some loss stats
            end_time = time.time()
            run_time = round(end_time - start_time)
            print(f'Epoch:[{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
                  f'Step:[{i+1:0>{len(str(len(encrypted_real_images)))}}/{len(encrypted_real_images)}]',
                  f'Loss-D:{d_loss.get_plain_text().item():.4f}',
                  f'Loss-G:{g_loss.get_plain_text().item():.4f}',
                #   f'D(x):{D_x:.4f}',
                #   f'D(G(z)):[{D_G_z1:.4f}/{D_G_z2:.4f}]',
                  f'Time:{run_time}s',
                  end='\r')
            
            G_losses.append(g_loss.get_plain_text().item())
            D_losses.append(d_loss.get_plain_text().item())

            # D_x_list.append(D_x)
            # D_z_list.append(D_G_z2)

            # 保存最好的模型
            if g_loss.get_plain_text() < loss_tep:
                torch.save(G.state_dict(), 'enc_model.pt')
                temp = g_loss
        
        with torch.no_grad():
            fake = G(fixed_noise).detach()
            fake = fake.get_plain_text()
        img_list.append(utils.make_grid(fake*0.5+0.5, nrow=10))
        img.extend(fake)  # 保存每张生成图像
        print()
        net_io_after = get_network_bytes()
        traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
        print("train:",traffic_diff)
        
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses[::100], label="G")
    plt.plot(D_losses[::100], label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.axhline(y=0, label="0", c="g")  # asymptote
    plt.legend()

    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
    ani = animation.ArtistAnimation(fig,
                                    ims,
                                    interval=1000,
                                    repeat_delay=1000,
                                    blit=True)
    HTML(ani.to_jshtml())

    # # Size of the Figure
    # plt.figure(figsize=(20, 10))
    # trainloader, testloader = preprocess_data()
    # # Plot the real images
    # plt.subplot(1, 2, 1)
    # plt.axis("off")
    # plt.title("Real Images")
    # real = next(iter(trainloader))
    # plt.imshow(
    #     utils.make_grid(real[0][:100] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))

    # # Load the Best Generative Model
    # netG = Generator()
    # netG.encrypt()
    # netG.load_state_dict(torch.load('enc_model.pt', map_location=torch.device('cpu')))
    # netG.eval()

    # # Generate the Fake Images
    # with torch.no_grad():
    #     fake = netG(fixed_noise.cpu())
    #     fake = fake.get_plain_text()

    # # Plot the fake images
    # plt.subplot(1, 2, 2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # fake = utils.make_grid(fake * 0.5 + 0.5, nrow=10)
    # plt.imshow(fake.permute(1, 2, 0))

    # Save the comparation result
    plt.savefig('lpgan_result.jpg', bbox_inches='tight')
    # 在训练循环结束后评估Inception Score
    mean_inception_score, std_inception_score = inception_score(img, inception_model)
    print(f"Inception Score: {mean_inception_score:.4f} ± {std_inception_score:.4f}")
    print_communication_costs()

run_lpgan()