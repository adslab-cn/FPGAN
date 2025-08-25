import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle

dataroot = "../data/MNIST"  # 数据集所在的路径，我们已经事先下载下来了
workers = 10  # 数据加载时的进程数
batch_size = 64  # 生成器输入的大小

image_size = 64  # 训练图像的大小
nc = 1  # 训练图像的通道数，彩色图像的话就是 3
nz = 100  # 输入是100 维的随机噪声 z，看作是 100 个 channel，每个特征图宽高是 1*1
ngf = 64  # 生成器中特征图的大小，
ndf = 64  # 判别器中特征图的大小
num_epochs = 5  # 训练的轮次
# num_epochs = 10  # 训练的轮次
lr = 0.0005  # 学习率大小
beta1 = 0.5  # Adam 优化器的参数
ngpu = 1  # 可用 GPU  的个数，0 代表使用 CPU

# 训练集加载并进行归一化等操作
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

# 测试集加载并进行归一化等操作
test_data = datasets.MNIST(root=dataroot,
                           train=False,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, ))
                           ]))

# 把 MNIST 的训练集和测试集都用来做训练
# dataset = train_data + test_data
dataset = train_data

from torch.utils.data import Subset
# 选择训练集的前 5000 张数据
subset_indices = list(range(5000))  # 创建索引范围
# dataset = Subset(train_data, subset_indices)

print(f'Total Size of Dataset: {len(dataset)}')

# 数据加载器，训练过程中不断产生数据
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=workers)

# 看是否存在可用的 GPU
device = torch.device('cpu')

# 权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# # new
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # 使用Xavier初始化卷积层  根据输入和输出神经元的数量动态调整权重的初始化范围
#         nn.init.xavier_uniform_(m.weight.data)
#         # nn.init.kaiming_normal_(m.weight, a=0.2)  # 对LeakyReLU特别有效
#     elif classname.find('BatchNorm') != -1:
#         # 使用标准正态分布初始化BatchNorm层
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.decon1 = nn.ConvTranspose2d(in_channels=nz, 
                                                 out_channels=ngf*8,
                                                 kernel_size=4, 
                                                 stride=1, 
                                                 padding=0,
                                                 bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.decon2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.decon3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.decon4 = nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=1, bias=False)  
        self.bn4 = nn.BatchNorm2d(ngf)
        self.decon5 = nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()

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
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc 
        self.ndf = ndf
        self.conv1 = nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)  # (64, 14, 14)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)  # (128, 7, 7)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)  # (256, 3, 3)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, 1, 1, stride=1, padding=0, bias=False)  # (1, 1, 1)
        # self.conv4 = nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=0, bias=False)  # (1, 1, 1)
        self.sd = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # print("conv1:", x)
        x = self.leaky_relu(x)
        # print("leaky_relu:", x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.sd(self.conv4(x))

        return x.view(x.size(0), -1).mean(dim=1)

# 创建一个生成器对象
netG = Generator(ngpu=0).to(device)
# 多卡并行，如果有多卡的话
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# # 初始化权重  其中，mean=0, stdev=0.2.
# netG.apply(weights_init)
for layer in netG.children():
    weights_init(layer)

# 创建一个判别器对象
netD = Discriminator(ngpu=0).to(device)
# 多卡并行，如果有多卡的话
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

# # 初始化权重  其中，mean=0, stdev=0.2.
# netD.apply(weights_init)
for layer in netD.children():
    weights_init(layer)

criterion = nn.BCELoss()  # 定义损失函数

# 创建一批潜在向量，我们将使用它们来可视化生成器的生成过程
fixed_noise = torch.randn(100, nz, 1, 1, device=device)

real_label = 1.  # “真”标签
fake_label = 0.  # “假”标签

optimizerG = torch.optim.SGD(netG.parameters(), lr=2e-2)
optimizerD = torch.optim.SGD(netD.parameters(), lr=2e-2)

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
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # calculate loss
    eps = 2 ** -16
    y_pred = torch.clamp(D_out.squeeze(), eps, 1 - eps)  # eps 是一个很小的值，如 1e-7
    loss = criterion(y_pred, labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # calculate loss
    eps = 2 ** -16
    y_pred = torch.clamp(D_out.squeeze(), eps, 1 - eps)  # eps 是一个很小的值，如 1e-7
    loss = criterion(y_pred, labels)
    # loss = criterion(D_out.squeeze(), labels)
    return loss

# 定义一个函数来计算梯度范数
def calculate_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # 计算L2范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  # 开平方得到总范数
    return total_norm

# 定义一些变量，用来存储每轮的相关值
img_list = []
img = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10

scaling_factor=1

# 训练并存储高精度数据
high_precision_data = []

print("Starting Training Loop...")
# 迭代
for epoch in range(num_epochs):
    beg_time = time.time()
    epoch_error = 0  # 初始化epoch误差
    # 数据加载器读取数据
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # 用所有的真数据进行训练
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        # print("data's size:", real_cpu.size())
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ),
                           real_label,
                           dtype=torch.float,
                           device=device)

        # 判别器推理
        # print(real_cpu)
        # print("b_size", b_size)

        output = netD(real_cpu).view(-1)
        # print("D_real_output:", output)
        # Calculate loss on all-real batch
        # 计算所有真标签的损失函数
        # errD_real = criterion(output, label)
        errD_real = real_loss(output, smooth=True)

        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_x = output.mean().item()

        # 生成假数据并进行训练
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        
        # 用生成器生成假图像
        g_time = time.time()
        fakeG = netG(noise)
        # print("G(z):", fakeG)
        g_end_time = time.time()
        # print("SG-FP:",fake)
        # print("Runtime\nSynthesis:", g_end_time-g_time)
        label.fill_(fake_label)
        # Classify all fake batch with D
        d_time = time.time()
        output = netD(fakeG.detach()).view(-1)
        # print("D_fake_output:", output)

        d_end_time = time.time()
        # print("Runtime\nInference:", d_end_time-d_time)

        # 计算判别器在假数据上的损失
        # errD_fake = criterion(output, label)
        # print("D(x1):", output)
        errD_fake = fake_loss(output)
        
        errD_fake.backward()

        # for param in netD.parameters():
        #     if param.grad is not None:
        #         param.grad *= scaling_factor

        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake 
        # errD = errD_real + errD_fake * 0.5 # 削弱对假样本的惩罚
        # errD.backward()
        
        original_d_grads = [param.grad.clone() for param in netD.parameters()]
        d_BP = []
        for item in original_d_grads:
            # print("item:",item.mean())
            d_BP.append(item)
        # print("SD-BP:", original_d_grads)
        # Update D
        optimizerD.step()
        d_grad_norm = calculate_gradient_norm(netD)
        # print(f"Discriminator Gradient Norm: {d_grad_norm:.4f}")
        # if step % 2 == 0:
        #     optimizerD.step()
        # optimizerG.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fakeG).view(-1)
        # print("D_real2_output:", output)

        # Calculate G's loss based on this output
        # errG = criterion(output, label)
        # print("D(x2):", output)
        errG = real_loss(output)
        # errG = real_loss(output) * 0.5 # 减弱生成器压力
        
        errG.backward()
        # # ====================
        # # 第二次训练Generator
        # # ====================
        # # 再次更新生成器
        # netG.zero_grad()
        # output = netD(fakeG).view(-1)  # 使用更新后的判别器
        # errG = real_loss(output)
        # errG.backward()  # 生成器的第二次反向传播
        # optimizerG.step()
        # # =========================
        
        # for param in netG.parameters():
        #     if param.grad is not None:
        #         param.grad *= scaling_factor


        original_g_grads = [param.grad.clone() for param in netG.parameters()]
        g_BP = []
        for item in original_d_grads:
            # print("item:",item.mean())
            g_BP.append(item)

        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        g_grad_norm = calculate_gradient_norm(netG)
        # print(f"Generator Gradient Norm: {g_grad_norm:.4f}")

        # Output training stats
        end_time = time.time()
        run_time = round(end_time - beg_time)
        print(f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
              f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
              f'Loss-D: {errD.item():.4f}',
              f'Loss-G: {errG.item():.4f}',
              f'D(x): {D_x:.4f}',
              f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
              f'Time: {run_time}s',
              end='\r')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        
        # Save D(X) and D(G(z)) for plotting later
        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)

        # 保存最好的模型
        if errG < loss_tep:
            torch.save(netG.state_dict(), 'modelG.pt')
            torch.save(netD.state_dict(), 'modelD.pt')
            temp = errG

    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_list.append(utils.make_grid(fake * 0.5 + 0.5, nrow=10))
    img.extend(fake)  # 保存每张生成图像
    print()

    # print("G-FP:",fakeG.mean())
    # print("D-FP:",output.mean())
    # print("G-BP:",g_BP)
    # print("D-BP:",d_BP)



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

# 在训练循环结束后评估Inception Score
mean_inception_score, std_inception_score = inception_score(img, inception_model)
print(f"Inception Score: {mean_inception_score:.4f} ± {std_inception_score:.4f}")

