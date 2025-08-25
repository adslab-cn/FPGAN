import torch
import torch.nn as nn
import crypten
import crypten.communicator as comm
import crypten.mpc as mpc

from crypten.lpgan.Secfunctions import print_communication_costs
crypten.init()

def construct_private_model(model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = crypten.nn.BatchNorm2d(num_features=256)
    private_model = model_upd.encrypt(src=0)
    return private_model

input_tensor = crypten.cryptensor(torch.randn(1, 256, 8, 8).to(torch.float64))  # (batch_size, num_channels, height, width)
# # 创建一个示例输入张量 (假设输入尺寸为 128×4×4) 这里的128表示特征图的数量，与 Batch Normalization 层的 num_features 参数一致
# input_tensor = crypten.cryptensor(torch.randn(1, 128, 4, 4).to(torch.float64))  # (batch_size, num_channels, height, width)
# bn_layer = crypten.nn.BatchNorm2d(num_features=128)
# bn_layer = crypten.nn.BatchNorm2d(num_features=256) 
# # bn_layer.encrypt()

# bn_layer = construct_private_model(bn_layer)
# # 前向传播
# output = bn_layer(input_tensor)
# output = net(input_tensor)

# sig = crypten.nn.Sigmoid()
# sig.encrypt()
# input_tensor = crypten.cryptensor(torch.randn(64, 32).to(torch.float64))  # (batch_size, num_channels, height, width)
# output = sig(input_tensor)


# conv = crypten.nn.Conv2d(in_channels=1, out_channels=28, kernel_size=5)
# conv.encrypt()
# input_tensor = crypten.cryptensor(torch.randn(1, 1, 28, 28).to(torch.float64))  # (batch_size, num_channels, height, width)
# # input_tensor = crypten.cryptensor(torch.randn(1, 3, 32, 32).to(torch.float64))  # (batch_size, num_channels, height, width)

# fc = crypten.nn.Linear(in_features=580, out_features=1)
# fc = crypten.nn.Linear(in_features=100, out_features=1)
# fc = crypten.nn.Linear(in_features=10, out_features=1)
# fc.encrypt()
# # 创建输入张量，假设输入尺寸为 (batch_size, features) = (100, 10)
# input_tensor = crypten.cryptensor(torch.randn(100, 10).to(torch.float64)) 
# input_tensor = crypten.cryptensor(torch.randn(784, 100).to(torch.float64)) 

@mpc.run_multiprocess(world_size=2)
def run():

    relu = crypten.nn.LeakyReLU(0.2)
    relu.encrypt()
    input_tensor = crypten.cryptensor(torch.randn(64, 32).to(torch.float64))  
    output = relu(input_tensor)
    print_communication_costs()
# loss = crypten.nn.BCEWithLogitsLoss()
# loss.encrypt()
# # # 64 个样本，每个样本的预测输出是一个 10 维的向量
# input_tensor = crypten.cryptensor(torch.randn(64, 10).to(torch.float64))  

# # BCEWithLogitsLoss 用于二分类任务，它通常期望标签是二进制值（0 或 1）
# labels = crypten.cryptensor(torch.randint(0, 2, (64, 10)).to(torch.float64))
# output = loss(input_tensor, labels)

# 打印输出的尺寸
run()
