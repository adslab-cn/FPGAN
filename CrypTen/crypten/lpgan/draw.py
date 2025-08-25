
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
from torchvision import models, transforms
import torch.nn.functional as F


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

netG = Generator()
netG.load_state_dict(torch.load('enc_model.pt', map_location=torch.device('cpu')))
netG.eval()

# Generate the Fake Images
with torch.no_grad():
    fake = netG(fixed_noise.cpu())
    fake = fake.get_plain_text()

# Plot the fake images
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
fake = utils.make_grid(fake * 0.5 + 0.5, nrow=10)
plt.imshow(fake.permute(1, 2, 0))

# Save the comparation result
plt.savefig('lpgan_result.jpg', bbox_inches='tight')
# 在训练循环结束后评估Inception Score
mean_inception_score, std_inception_score = inception_score(img, inception_model)
print(f"Inception Score: {mean_inception_score:.4f} ± {std_inception_score:.4f}")