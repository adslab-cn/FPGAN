# import the libraries
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.aux_parameter import DCFKey
from NssMPC import RingTensor
import random
import torch
import crypten
import crypten.communicator as comm
import torch.distributed as dist
from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret, print_communication_costs, print_execution_times
from crypten.lpgan.Secfunctions import SecAnd, SecOr, get_network_bytes, calculate_network_traffic

num_of_keys = 4  # We need a few keys for a few function values, but of course we can generate many keys in advance.

# generate keys in offline phase
# set alpha and beta
r = 9
# r = random.randint(0, 100)
alpha = RingTensor.convert_to_ring(r)


import os
os.environ["GLOO_SOCKET_IFNAME"] = 'enp3s0'
os.environ["GLOO_IPV6"] = '0'


crypten.init()

def selective_clamp(x, a, b, eps):
    in_range_mask = x.ibdicf()
    return x * in_range_mask + a * (1 - in_range_mask) * a.relu() + b * (1 - in_range_mask) * crypten.greater(x, b)

def compute_bceloss(input_shape, target_shape=None):
    """计算给定输入形状的 BCELoss"""
    if target_shape is None:
        target_shape = input_shape  # 默认目标形状与输入相同

    # 生成输入（模拟经过 sigmoid 的输出）
    x = torch.randn(input_shape)
    # x = torch.tensor([2,2])
    x = torch.sigmoid(x)  # 确保输入在 [0, 1] 范围内
    x_enc = crypten.cryptensor(x, ptype=crypten.mpc.arithmetic)

    # 生成目标值（0 或 1）
    target = torch.randint(0, 2, target_shape).float()
    target_enc = crypten.cryptensor(target, ptype=crypten.mpc.arithmetic)

    # 计算 BCELoss
    loss_BCE = crypten.nn.BCELoss()
    loss_ESBCE = crypten.nn.ESBCELoss()
    loss_torch = torch.nn.BCELoss()
    loss = loss_torch(x, target)
    lossB = loss_BCE(x_enc, target_enc)
    lossE = loss_ESBCE(x_enc, target_enc)

    # 解密并返回结果
    return lossB.get_plain_text(), lossE.get_plain_text(), loss

import crypten.mpc as mpc

def run():

    # 测试 [64, 32]
    # lossB_64x32, lossE_64x32, loss= compute_bceloss([2])
    # lossB_64x32, lossE_64x32, loss= compute_bceloss([64,32])
    # lossB_64x32, lossE_64x32, loss= compute_bceloss([256, 20])
    lossB_64x32, lossE_64x32, loss= compute_bceloss([169, 169])
    print(f"BCELoss for [64, 32]: {lossB_64x32},\n{lossE_64x32},\n{loss}")

    # # 测试 [256, 20]
    # loss_256x20 = compute_bceloss([256, 20])
    # print(f"BCELoss for [256, 20]: {loss_256x20}")

    # # 测试 [169, 169]
    # loss_169x169 = compute_bceloss([169, 169])
    # print(f"BCELoss for [169, 169]: {loss_169x169}")

    # x = crypten.cryptensor([[[[-0.3524, 0.2244, 9, 10, 7]]]])
    # x = crypten.cryptensor([[[[-6.0131e-03, 0.42562, 1.9, 10, 7]]]])
    # x = crypten.cryptensor([[[[1, 3, 9, 4, 7]]]])

    # x = crypten.cryptensor(torch.randn([3,3,169,169]), ptype=crypten.mpc.arithmetic)

    # x = crypten.cryptensor(torch.randn([64,32]), ptype=crypten.mpc.arithmetic)
    # x = crypten.cryptensor(torch.randn([256,20]), ptype=crypten.mpc.arithmetic)
    # x = crypten.cryptensor(torch.randn([169,169]), ptype=crypten.mpc.arithmetic)

    # z = crypten.nn.BatchNorm2d(32)
    # z.encrypt()
    # y = z(x)
    # print(y.get_plain_text())
        
    # y = x.fastleakyrelu()
    # print("x.fastleakyrelu():", y.get_plain_text())

    # y = x.fastsecnet()
    # print("x.fastsecnet():", y.get_plain_text())

    # y = x.relu()
    # print("x.funshade():", y.get_plain_text())

    # y = x.fss()
    # print("x.funshade():", y)

    # y = x.cmp()
    # print("x.cmp():", y)
    
    # y = x.nss_leakyrelu()
    # y = x.LPGAN_leakyrelu()
    # print("x.LPGAN_leakyrelu():", (y))

    # y = x.CrypTen_leakyrelu()
    # x.dcf()
    # x.ibdcf(alpha=3) # right
    
    # y = x.ibdicf()
    # print("ibdicf:", y)
    # x.dicf()


    # x.fss()
    # x.cmp()
 
    
    # y = x.relu()
    # print("x.relu():", (y.get_plain_text()))


    print_communication_costs()
    print_execution_times()

run()



# def generate_shares(secret, modulus=2**16):
#     """
#     Perform two-party secret sharing in the ring of size modulus (2^16).
#     The secret is split into two shares, `share1` and `share2`, within the modulus.
#     """
#     if isinstance(secret, (int, float)):
#         # 对于 int 和 float 类型，先转换为整数，并将其限制在 [0, modulus-1] 范围内
#         secret = int(secret) % modulus
        
#         # 生成两个随机值，限制在 [0, modulus-1] 范围内
#         share1 = random.randint(0, modulus - 1)
#         share2 = (secret - share1) % modulus  # 确保 share1 + share2 ≡ secret (mod modulus)
#         return share1, share2

#     elif isinstance(secret, list):
#         # 对于 list 类型，我们对其中的每个元素进行秘密分享
#         secret = [int(x) % modulus for x in secret]  # 确保每个元素都在 [0, modulus-1] 范围内
#         share1 = [random.randint(0, modulus - 1) for _ in secret]
#         share2 = [(s - sh) % modulus for s, sh in zip(secret, share1)]  # 确保 share1 + share2 ≡ secret (mod modulus)
#         return share1, share2

#     else:
#         raise ValueError(f"Unsupported type for secret sharing: {type(secret)}")
