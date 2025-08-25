#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
from crypten.common.tensor_types import is_tensor
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.aux_parameter import DCFKey
from NssMPC import RingTensor
from NssMPC.crypto.primitives.function_secret_sharing.dicf import DICF, ibDICF, ibDCF
from NssMPC.crypto.aux_parameter import DICFKey

# import funshade
import numpy as np
import torch
# import crypten
import crypten.communicator as comm
import torch.distributed as dist
import crypten.mpc as mpc
from crypten.lpgan.Secfunctions import track_network_traffic, timeit

__all__ = [
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ne__",
    "abs",
    "eq",
    "ge",
    "gt",
    "hardtanh",
    "le",
    "lt",
    "ne",
    "relu",
    "dcf",
    "ibdcf",
    "ibdicf",
    "fastleakyrelu",
    "nss",
    "dicf",
    "fastsecnet",
    "fss",
    "cmp",
    "nss_leakyrelu",
    "LPGAN_leakyrelu",
    "CrypTen_leakyrelu",
    # "secrelu",
    "sign",
    "where",
]


def ge(self, y):
    """Returns self >= y"""
    return 1 - self.lt(y)


def gt(self, y):
    """Returns self > y"""
    return (-self + y)._ltz()


def le(self, y):
    """Returns self <= y"""
    return 1 - self.gt(y)


def lt(self, y):
    """Returns self < y"""
    return (self - y)._ltz()


def eq(self, y):
    """Returns self == y"""
    return 1 - self.ne(y)


def ne(self, y):
    """Returns self != y"""
    difference = self - y
    difference = type(difference).stack([difference, -difference])
    return difference._ltz().sum(0)


__eq__ = eq
__ge__ = ge
__gt__ = gt
__le__ = le
__lt__ = lt
__ne__ = ne


def sign(self):
    """Computes the sign value of a tensor (0 is considered positive)"""
    return 1 - 2 * self._ltz()


def abs(self):
    """Computes the absolute value of a tensor"""
    return self * self.sign()
    
@timeit
@track_network_traffic
def cmp(u_shared):
    import random
    import crypten.mpc
    import torch
    import crypten
    import crypten.mpc as mpc
    import torch.distributed as dist
    import crypten.communicator as comm
    from crypten.mpc.primitives import BinarySharedTensor
    from crypten.lpgan import DFC
    from crypten.lpgan import SecMSB
    from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
    from crypten.mpc.provider.ttp_provider import TTPClient
    from crypten.lpgan.Secfunctions import SecAnd, SecOr, get_network_bytes, calculate_network_traffic,pack_bits,unpack_bits
    import time
    u_shared = u_shared._tensor.data
    v_shared = crypten.cryptensor(0)
    v_shared = v_shared._tensor.data
    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()

    print("=============cmp")
    # 1)2) a=u1-v1,b=v2-u2
    if rank == 0:
        a = u_shared - v_shared

    elif rank == 1:
        b = v_shared - u_shared


    dist.barrier()
    crypten.print(f"Rank {rank} barrier 1 passed")

    # net_io_before = get_network_bytes()
    if rank == 0:
        a_signed_bit, a_unsigned_bit = DFC.DFC(a)
        a_bit = a_unsigned_bit
        r0, s0 = provider.generate_shares(a_bit)
        s1 = torch.zeros_like(s0)
        s0 = pack_bits(s0)
        
        a_sign = DFC.extract_sign_bits(a_signed_bit)
        b_sign = torch.zeros_like(a_sign)
    elif rank == 1:
        b_signed_bit, b_unsigned_bit = DFC.DFC(b)
        b_bit = b_unsigned_bit
        r1, s1 = provider.generate_shares(b_bit)
        s0 = torch.zeros_like(s1)
        s1 = pack_bits(s1)

        b_sign = DFC.extract_sign_bits(b_signed_bit)
        a_sign = torch.zeros_like(b_sign)

    dist.barrier()
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)
    # crypten.print(f"Rank {rank} barrier 2 passed")
    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecXor:",traffic_diff)

    # 3) SecXor(a_bit, b_bit)
    if rank == 0:
        s1 = unpack_bits(s1,r0.shape)
        c_bit = r0 ^ s1
    elif rank == 1:
        s0 = unpack_bits(s0,r1.shape)
        c_bit = r1 ^ s0
    dist.barrier()


    # 4) SecMSB(c_bit', c_bit'')
    # net_io_before = get_network_bytes()
    d_bit = SecMSB.SecMSB(c_bit)
    
    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecMSB:",traffic_diff)

    # net_io_before = get_network_bytes()
    # 5) e_bit = SecAnd(a_bit, d_bit)
    # a_bit from S0, d_bit from S0 and S1
    a, b, c = provider.generate_binary_triple(d_bit.size(), d_bit.size(), device=d_bit.device)
    a = a._tensor.data.bool()
    b = b._tensor.data.bool()
    c = c._tensor.data.bool()

    if rank == 0:
        e_local = (r0 ^ a)
        f_local = (d_bit ^ b)

    elif rank == 1:
        e_local = (s0 ^ a)
        f_local = (d_bit ^ b)
        
    e = pack_bits(e_local)
    f = pack_bits(f_local)
    
    # Step 2: Gather e and f from both parties
    e_list = [torch.zeros_like(e) for _ in range(2)]
    f_list = [torch.zeros_like(f) for _ in range(2)]
    dist.all_gather(e_list, e)
    dist.all_gather(f_list, f)

    # if rank == 0:
    #     # Party 0 发送 e_local 和 f_local 给 Party 1
    #     comm.get().send(e, dst=1)
    #     comm.get().send(f, dst=1)
    #     # Party 0 直接使用本地的 e_local 和 f_local
    #     e, f = e, f
    # else:
    #     # Party 1 接收 e 和 f
    #     e = torch.zeros_like(e)
    #     f = torch.zeros_like(f)
    #     comm.get().recv(e, src=0)
    #     comm.get().recv(f, src=0)

    # Step 3: Compute e and f
    e = e_list[0] ^ e_list[1]
    f = f_list[0] ^ f_list[1]

    e = unpack_bits(e,e_local.shape)
    f = unpack_bits(f,f_local.shape)
    
    if rank == 0:
        z0 = c ^ a&f ^ b&e ^ e&f
        e_bit = z0


    elif rank == 1:
        z1 = c ^ a&f ^ b&e
        e_bit = z1
    dist.barrier()
    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecAnd:",traffic_diff)

    # net_io_before = get_network_bytes()
    # 7) SecOr(a_sign, b_sign)

    lota = SecOr(a_sign, b_sign)


    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecOr:",traffic_diff)

    # 6) 
    # Xi = e_bit.sum(dim=-1) 
    # zeta = d_bit.sum(dim=-1)

    sum_row = torch.sum(d_bit, dim=-1, keepdim=True)  # 按行求和
    zeta_ = (sum_row % 2).to(torch.bool).squeeze(-1)  # 取模2并转换类型

    sum_row = torch.sum(e_bit, dim=-1, keepdim=True)  # 按行求和
    Xi_ = (sum_row % 2).to(torch.bool).squeeze(-1)  # 取模2并转换类型
    

    # 8) v = SecXor(lota, Xi)
    # net_io_before = get_network_bytes()
    v = lota ^ Xi_ # if a,b<0 Xi=0 lota=1

    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecXor:",traffic_diff)

    # 9) SecAnd(v, zeta)
    # net_io_before = get_network_bytes()
    f = SecAnd(v, zeta_)
    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_SecAnd:",traffic_diff)
    
    if rank == 0:
        z0 = f
        z1 = torch.zeros_like(f)
    else:
        z1 = f
        z0 = torch.zeros_like(z1)
    dist.barrier()
    dist.broadcast(z0, src=0)
    dist.broadcast(z1, src=1)
    f = z0 ^ z1
    # print(f"cmp:{f}")
    return f



# K = 1       # Number of elements in the input vector
theta = 0    # Threshold to compare it with
@track_network_traffic
def fss(a):
    # Create integer vector of length K
    # K = (a.size())[-1]
    K = 1
    for dim in a.size():
        K *= dim
    rng = np.random.default_rng(seed=42)
    a = a  # Tensor a
    # z = crypten.cryptensor(torch.tensor([a], dtype=torch.int64))  # Input vector z
    z = a._tensor.data

    ## Offline Phase
    r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, theta)
    r_in0 = torch.tensor(r_in0)
    r_in1 = torch.tensor(r_in1)
    z = z.view(-1) 

    ## Online Phase
    net_io_before = get_network_bytes()
    rank = comm.get().get_rank()
    if rank == 0:
        z_0 = z.to('cpu') + r_in0
        z_0 = z_0.numpy().astype(np.int32)
        z_0 = torch.tensor(z_0)  # Convert to torch.Tensor
        z_1 = torch.zeros_like(z_0)  # Initialize z_1 to zeros
    else:
        z_1 = z.to('cpu') + r_in1
        z_1 = z_1.numpy().astype(np.int32)
        z_1 = torch.tensor(z_1)  # Convert to torch.Tensor
        z_0 = torch.zeros_like(z_1)  # Initialize z_0 to zeros
    dist.barrier()
    dist.broadcast(z_0, src=0)
    dist.broadcast(z_1, src=1)
    z_0 = z_0.view(-1)
    z_1 = z_1.view(-1)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_z.get_plain_text:",traffic_diff)

    # Compute the comparison to the threshold theta
    if rank == 0:
        o_j_1 = funshade.eval_sign(K, rank, k0, z_0.numpy(), z_1.numpy())
        o_j_1 = torch.tensor(o_j_1)  # Convert to torch.Tensor
        o_j_2 = torch.zeros_like(o_j_1)
    else:
        o_j_2 = funshade.eval_sign(K, rank, k1, z_1.numpy(), z_0.numpy())
        o_j_2 = torch.tensor(o_j_2)  # Convert to torch.Tensor
        o_j_1 = torch.zeros_like(o_j_2)

    dist.barrier()
    dist.broadcast(o_j_1, src=0)
    dist.broadcast(o_j_2, src=1)

    o = o_j_1 + o_j_2
    return o

@timeit
@track_network_traffic
def fastsecnet(u):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    import torch
    import crypten

    # 定义缩放因子Q，这里选择2^20以保持精度并避免溢出
    Q = 2^20
    # 将输入小数缩放为整数
    u_scaled = (u * Q)  # 四舍五入并转换为整数

    # 调整掩码参数r为Q，确保x_scaled + r始终非负
    r = 2**16
    alpha = RingTensor.convert_to_ring(r)

    # 生成共享的随机值（根据新的r调整）
    b00, b01 = generate_share(1)
    b10, b11 = generate_share(-r)

    r0, r1 = generate_share(r)
    beta = RingTensor.convert_to_ring([1, -r])

    # ======================
    #     online phase
    # ======================
    # 将缩放后的输入移到GPU并加密
    u = u_scaled  # 加密缩放后的整数
    
    v = crypten.cryptensor(r)
    y = u + v  # 安全加法

    # 计算密钥数量K（保持原逻辑）
    K = 1
    for dim in u.size():
        K *= dim
    Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=-beta)

    # 分发密钥
    k0 = [Key0, r0, b00, b10]
    k1 = [Key1, r1, b01, b11]

    # 通信并解密y（注意处理缩放后的整数）
    y = y.get_plain_text().long()  # 确保结果为整数类型

    x = RingTensor.convert_to_ring(y)
    rank = comm.get().get_rank()
    
    # 分布式评估
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)
    else:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    dist.barrier()

    # 处理维度
    x = x.unsqueeze(1)
    b0 = torch.tensor([k0[2], k0[3]])
    b1 = torch.tensor([k1[2], k1[3]])

    # 各参与方计算局部结果
    if rank == 0:
        b0p = (res_0.tensor + b0)
        z0 = elementwise_multiply(b0p[:,0].reshape(x.shape), x)
        y0 = elementwise_add(z0, b0p[:,1].reshape(x.shape))
        y0 = y0.squeeze(1)
        y1 = torch.zeros_like(y0)
    else:
        b1p = (res_1.tensor + b1)
        z1 = elementwise_multiply(b1p[:,0].reshape(x.shape), x)
        y1 = elementwise_add(z1, b1p[:,1].reshape(x.shape))
        y1 = y1.squeeze(1)
        y0 = torch.zeros_like(y1)

    # 同步结果并恢复秘密
    dist.barrier()
    dist.broadcast(y0, src=0)
    dist.broadcast(y1, src=1)
    y = recover_secret(y0, y1)
    
    # 将结果缩放回小数
    # y = y.float() / Q  # 转换为浮点数并除以Q
    y = y / Q  # 转换为浮点数并除以Q
    
    y = crypten.cryptensor(y)
    print("======fastsecnet=====")
    return y.cpu()

# def fastsecnet(u):
#     from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
#     # fastsecnet的思想：x+r<r, 其实就是x<0, 那只要让r足够大，我能涵盖的负数就越大
#     r = 2**16

#     # r = random.randint(0, 100)
#     alpha = RingTensor.convert_to_ring(r)

#     b00, b01 = generate_share(1)
#     b10, b11 = generate_share(-r)

#     r0, r1 = generate_share(r)

#     beta = RingTensor.convert_to_ring([1,-r])
#     # ======================
#     #     online phase
#     # ======================


#     v = crypten.cryptensor(r)
#     y = u + v

#     K = 1
#     for dim in u.size():
#         K *= dim
#     Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=-beta)

#     k0 = [Key0, r0, b00, b10]
#     k1 = [Key1, r1, b01, b11]

#     # 唯一的一次通信
#     y = y.get_plain_text().int()

#     x = RingTensor.convert_to_ring(y)
#     rank = comm.get().get_rank()
#     r_0 = DCF.eval(x=x, keys=Key0, party_id=0)
#     r_1 = DCF.eval(x=x, keys=Key1, party_id=1)
#     res = r_0 + r_1
#     # Party 0:
#     if rank == 0:
#         res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

#     # Party 1:
#     if rank == 1:
#         res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

#     dist.barrier()

#     x = x.unsqueeze(1)  
#     b0 = torch.tensor([k0[2], k0[3]])
#     b1 = torch.tensor([k1[2], k1[3]])

#     # b = recover_secret(b0,b1)

#     sb0p = (r_0.tensor + b0)
#     sb1p = (r_1.tensor + b1)
#     if rank == 0:
#         b0p = (res_0.tensor + b0)
#         z0 = elementwise_multiply(b0p[:,0].reshape(x.shape), x)
#         y0 =  elementwise_add(z0, b0p[:,1].reshape(x.shape))

#         y0 = y0.squeeze(1)
#         y1 = torch.zeros_like(y0)

#         # return y0
#     else:
#         b1p = (res_1.tensor + b1)
#         z1 = elementwise_multiply(b1p[:,0].reshape(x.shape), x)
#         y1 =  elementwise_add(z1, b1p[:,1].reshape(x.shape))
#         y1 = y1.squeeze(1)
#         y0 = torch.zeros_like(y1)
   
#         # return y1
#     dist.barrier()
#     dist.broadcast(y0, src=0)
#     dist.broadcast(y1, src=1)
#     y = recover_secret(y0,y1)
    
#     y = crypten.cryptensor(y)
#     print("======fastsecnet=====")
#     return y.cpu()



from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
import torch
import torch.distributed as dist

@timeit
@track_network_traffic
def fastleakyrelu(u, a=0.02):
    r = 2**16
    scale = 100000

    u = u*scale
    alpha = RingTensor.convert_to_ring(r)
    # print("===========fast============")
    b00, b01 = generate_share(1)
    b10, b11 = generate_share(-a * r)
    b20, b21 = generate_share(-r)

    r0, r1 = generate_share(r)
    beta = RingTensor.convert_to_ring([1, -a * r, -r])
    mask = torch.tensor([0, 0, 1])
    size = u.shape[-1]

    mask = mask.unsqueeze(0).expand(size, -1)  # 扩展为 [r, 3]

    # ======================
    #     online phase
    # ======================

    v = crypten.cryptensor(r)
    y = u + v
    K = 1
    for dim in u.size():
        K *= dim

    Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=beta)

    k0 = [Key0, r0, b00, b10, b20]
    k1 = [Key1, r1, b01, b11, b21]

    rank = comm.get().get_rank()
    # 唯一的一次通信
    net_io_before = get_network_bytes()
    y = y.get_plain_text().int()

    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_y.get_plain_text:",traffic_diff)

    
    x = RingTensor.convert_to_ring(y.int())


    # Party 0:
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)


    dist.barrier()

    b0 = torch.tensor([k0[2], k0[3], k0[1]])
    b1 = torch.tensor([k1[2], k1[3], k1[1]])

    a = torch.tensor(a)
    a = torch.tensor(a * scale).to(torch.int64)

    if rank == 0:
        # 计算需要重复的次数
        repeat_times = res_0.tensor.size(0) // mask.size(0)
        mask = torch.tile(mask, (repeat_times, 1))

        if res_0.tensor.size(0) % mask.size(0) != 0:
            remainder = res_0.tensor.size(0) % mask.size(0)
            mask = torch.cat([mask, mask[:remainder]], dim=0)

        # 4) compute b0p
        b0p = res_0.tensor + b0[2] * mask * 65536

        c = torch.tensor(a * x.tensor).to(torch.int64)

        z1 = b0p[:, 0].reshape(c.shape) * c
        z2 = b0p[:, 1].reshape(c.shape) * scale
        c3 = b0[0] - b0p[:, 0].reshape(c.shape)
        z3 = elementwise_multiply(c3, x.tensor) * scale
        z4 = b0p[:, 2].reshape(c.shape) * scale
            
        y0 = z1 + z2 + z3 - z4
        # return y0

        y1 = torch.zeros_like(y0)

    else:
        repeat_times = res_1.tensor.size(0) // mask.size(0)
        mask = torch.tile(mask, (repeat_times, 1))

        if res_1.tensor.size(0) % mask.size(0) != 0:
            remainder = res_1.tensor.size(0) % mask.size(0)
            mask = torch.cat([mask, mask[:remainder]], dim=0)

        b1p = res_1.tensor + b1[2] * mask * 65536
        c = torch.tensor((a * x.tensor)).to(torch.int64)


        z1 = b1p[:, 0].reshape(c.shape) * c
        z2 = b1p[:, 1].reshape(c.shape) * scale
        c3 = b1[0] - b1p[:, 0].reshape(c.shape)
        z3 = elementwise_multiply(c3, x.tensor) * scale
        
            
        z4 = b1p[:, 2].reshape(c.shape) * scale
        y1 = z1 + z2 + z3 - z4
        # return y1

        y0 = torch.zeros_like(y1)

    # 使用 all_reduce 合并数据
    combined = y0 if rank == 0 else y1
    dist.all_reduce(combined, op=dist.ReduceOp.SUM)

    # 计算结果
    y = combined / 65536 / scale / scale

    y = crypten.cryptensor(y)
    return y.cpu()

# @timeit
# @track_network_traffic
# def fastleakyrelu(u, a=0.02):
#     r = 2**16
#     u = u
#     alpha = RingTensor.convert_to_ring(r)
#     # print("===========fast============")
#     b00, b01 = generate_share(1)
#     b10, b11 = generate_share(-a * r)
#     b20, b21 = generate_share(-r)

#     r0, r1 = generate_share(r)
#     beta = RingTensor.convert_to_ring([1, -a * r, -r])
#     mask = torch.tensor([0, 0, 1])
#     size = u.shape[-1]

#     mask = mask.unsqueeze(0).expand(size, -1)  # 扩展为 [r, 3]

#     # ======================
#     #     online phase
#     # ======================

#     v = crypten.cryptensor(r)
#     y = u + v
#     K = 1
#     for dim in u.size():
#         K *= dim

#     Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=beta)

#     k0 = [Key0, r0, b00, b10, b20]
#     k1 = [Key1, r1, b01, b11, b21]

#     rank = comm.get().get_rank()
#     # 唯一的一次通信
#     net_io_before = get_network_bytes()
#     y = y.get_plain_text().int()

#     net_io_after = get_network_bytes()
#     traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
#     # crypten.print("traffic_y.get_plain_text:",traffic_diff)

    
#     x = RingTensor.convert_to_ring(y.int())


#     # Party 0:
#     if rank == 0:
#         res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

#     # Party 1:
#     if rank == 1:
#         res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

#     dist.barrier()

#     b0 = torch.tensor([k0[2], k0[3], k0[1]])
#     b1 = torch.tensor([k1[2], k1[3], k1[1]])

#     a = torch.tensor(a)
#     scale = 1000
#     a = torch.tensor(a * scale).to(torch.int64)

#     if rank == 0:
#         # 计算需要重复的次数
#         repeat_times = res_0.tensor.size(0) // mask.size(0)
#         mask = torch.tile(mask, (repeat_times, 1))

#         if res_0.tensor.size(0) % mask.size(0) != 0:
#             remainder = res_0.tensor.size(0) % mask.size(0)
#             mask = torch.cat([mask, mask[:remainder]], dim=0)

#         # 4) compute b0p
#         b0p = res_0.tensor + b0[2] * mask * 65536

#         c = torch.tensor(a * x.tensor).to(torch.int64)
        
#         z1 = b0p[:, 0].reshape(c.shape) * c
#         z2 = b0p[:, 1].reshape(c.shape) * scale
#         c3 = b0[0] - b0p[:, 0].reshape(c.shape)
#         z3 = elementwise_multiply(c3, x.tensor) * scale
#         z4 = b0p[:, 2].reshape(c.shape) * scale
            
#         y0 = z1 + z2 + z3 - z4
#         signs = torch.sign(y0)
#         last_10_digits = torch.abs(y0) % 10**10
#         y0 = last_10_digits * signs.to(torch.float64) / 1000

        
#         # y0 = y0[0]
#         y1 = torch.zeros_like(y0)

#     else:
#         repeat_times = res_1.tensor.size(0) // mask.size(0)
#         mask = torch.tile(mask, (repeat_times, 1))

#         if res_1.tensor.size(0) % mask.size(0) != 0:
#             remainder = res_1.tensor.size(0) % mask.size(0)
#             mask = torch.cat([mask, mask[:remainder]], dim=0)

#         b1p = res_1.tensor + b1[2] * mask * 65536
#         c = torch.tensor((a * x.tensor)).to(torch.int64)

#         z1 = b1p[:, 0].reshape(c.shape) * c
#         z2 = b1p[:, 1].reshape(c.shape) * scale
#         c3 = b1[0] - b1p[:, 0].reshape(c.shape)
#         z3 = elementwise_multiply(c3, x.tensor) * scale
            
#         z4 = b1p[:, 2].reshape(c.shape) * scale
#         y1 = z1 + z2 + z3 - z4
#         signs = torch.sign(y1)
#         last_10_digits = torch.abs(y1) % 10**10
#         y1 = last_10_digits * signs.to(torch.float64) / 1000
#         # y1 = y1[0]
#         y0 = torch.zeros_like(y1)

#     # 使用 all_reduce 合并数据
#     combined = y0 if rank == 0 else y1
#     dist.all_reduce(combined, op=dist.ReduceOp.SUM)

#     # 计算结果
#     y = combined / 65536

#     y = crypten.cryptensor(y)
#     return y

@track_network_traffic
def nss(u):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    r = 2**16
    u = u
    # r = random.randint(0, 100)
    alpha = RingTensor.convert_to_ring(0)


    beta = RingTensor.convert_to_ring([0])
    # ======================
    #     online phase
    # ======================
    # generate some values what we need to evaluate
    # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    # u = crypten.cryptensor(x)
    v = crypten.cryptensor(r)
    y = u + v

    K = 1
    for dim in u.size():
        K *= dim

    Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=beta)


    # 唯一的一次通信
    y = y.get_plain_text().int()

    x = RingTensor.convert_to_ring(y)
    rank = comm.get().get_rank()
    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)
        res_0 = res_0.tensor
        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)
        res_1 = res_1.tensor
        res_0 = torch.zeros_like(res_1)

    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)
    res = res_0 + res_1
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_nss:",traffic_diff)
    # print("res:", res)
    return res

@timeit
@track_network_traffic
def dicf(x, p=0, q=2**16):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    # r = random.randint(0, 100)
    down_bound = RingTensor.convert_to_ring(p)
    upper_bound = RingTensor.convert_to_ring(q)
    # ======================
    #     online phase
    # ======================
    # generate some values what we need to evaluate
    # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    K = 1
    for dim in x.size():
        K *= dim

    Key0, Key1 = DICFKey.gen(num_of_keys=K, down_bound=down_bound, upper_bound=upper_bound)
    
    x = RingTensor(x.get_plain_text().to(torch.int64))
    x_shift = x + Key0.r_in.reshape(x.shape) + Key1.r_in.reshape(x.shape)

    # x_shift = x_shift.to(torch.int64)
    rank = comm.get().get_rank()
    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = DICF.eval(x_shift=x_shift, keys=Key0, party_id=0, down_bound=down_bound, upper_bound=upper_bound)
        res_0 = res_0.tensor
        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = DICF.eval(x_shift=x_shift, keys=Key1, party_id=1, down_bound=down_bound, upper_bound=upper_bound)
        res_1 = res_1.tensor
        res_0 = torch.zeros_like(res_1)

    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)
    res = res_0 + res_1
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_dicf:",traffic_diff)
    print("dicf:", res)
    return res
    
@timeit
@track_network_traffic
def ibdicf(x, p=0, q=2**16):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    x = x
    # r = random.randint(0, 100)
    down_bound = RingTensor.convert_to_ring(p)
    upper_bound = RingTensor.convert_to_ring(q)

    # ======================
    #     online phase
    # ======================
    # generate some values what we need to evaluate
    # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    K = 1
    for dim in x.size():
        K *= dim

    Key0, Key1 = ibDICF.gen(num_of_keys=K, down_bound=down_bound, upper_bound=upper_bound)
    
    x = RingTensor(x.get_plain_text().to(torch.int64))
    x_shift = x + Key0.r_in.reshape(x.shape) + Key1.r_in.reshape(x.shape)

    # x_shift = x_shift.to(torch.int64)
    rank = comm.get().get_rank()
    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = ibDICF.eval(x_shift=x_shift, keys=Key0, party_id=0, down_bound=down_bound, upper_bound=upper_bound)

        res_0 = res_0.tensor
        res_0 = res_0.to(torch.uint8)

        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = ibDICF.eval(x_shift=x_shift, keys=Key1, party_id=1, down_bound=down_bound, upper_bound=upper_bound)

        res_1 = res_1.tensor
        # print("====in ibdicf:\nres_1", res_1)
        res_1 = res_1.to(torch.uint8)

        res_0 = torch.zeros_like(res_1)

    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)
    # print("=========in ibdicf's res_1:", res_1)
    # dist.broadcast(res_0[:,0].reshape(x.shape), src=0)
    # dist.broadcast(res_1[:,0].reshape(x.shape), src=1)
    # print("=======in ibdicf\nres_0[:, 0]", res_0[:, 0].size())
    res = res_0 + res_1


    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    res = torch.where(res.abs() == 1, res.abs(), torch.tensor(0))
    # print("ibdicf:", res[:, -1])

    return res[:, -1]


@track_network_traffic
def ibdcf(x, alpha=1):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    x = x
    alpha = RingTensor.convert_to_ring(alpha)
    beta = RingTensor(1)

    # ======================
    #     online phase
    # ======================
    # generate some values what we need to evaluate
    # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    K = 1
    for dim in x.size():
        K *= dim

    Key0, Key1 = ibDCF.gen(num_of_keys=K, alpha=alpha, beta=beta)
    
    x = RingTensor(x.get_plain_text().to(torch.int64))
    # x_shift = x + Key0.r_in.reshape(x.shape) + Key1.r_in.reshape(x.shape)

    # x_shift = x_shift.to(torch.int64)
    rank = comm.get().get_rank()
    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = ibDCF.eval(x=x, keys=Key0, party_id=0)
        res_0 = res_0[:,0:1]
        res_0 = res_0.to(torch.uint8)
        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = ibDCF.eval(x=x, keys=Key1, party_id=1)
        res_1 = res_1[:,0:1]
        res_1 = res_1.to(torch.uint8)

        res_0 = torch.zeros_like(res_1)
    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)

    res = res_0 ^ res_1

    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # crypten.print("traffic_ibdcf:",traffic_diff)
    # print("ibdcf:", res)

    # print("ibdicf:", res)
    return res

@track_network_traffic
def dcf(x, alpha=1):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    x = x
    alpha = RingTensor.convert_to_ring(alpha)
    beta = RingTensor(1)

    # ======================
    #     online phase
    # ======================
    # generate some values what we need to evaluate
    # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    K = 1
    for dim in x.size():
        K *= dim

    Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=beta)
    
    x = RingTensor(x.get_plain_text().to(torch.int64))
    # x_shift = x + Key0.r_in.reshape(x.shape) + Key1.r_in.reshape(x.shape)

    # x_shift = x_shift.to(torch.int64)
    rank = comm.get().get_rank()
    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)
        res_0 = res_0.tensor

        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

        res_1 = res_1.tensor

        res_0 = torch.zeros_like(res_1)

    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)


    res = res_0 + res_1


    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_dcf:",traffic_diff)

    print("dcf:", res/65536)
    return res


from crypten.lpgan.Secfunctions import timeit, track_network_traffic,get_network_bytes, calculate_network_traffic
@timeit
@track_network_traffic
def relu(self):
    # net_io_before = get_network_bytes()
    # # y = self.view(-1).to('cpu') * self.to('cpu').fss()

    # # y = self.view(-1) * (self.nss()).view(-1)
    # # y = self.view(-1) * ((self.ibdicf())[:, 0]).view(-1).to('cpu')
    # # y = y.get_plain_text()
    # net_io_after = get_network_bytes()
    # traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # # crypten.print("traffic_relu:",traffic_diff)
    # # print("y:", y)

    
    # ===============
    # y = self.fastsecnet()
    # return y
    # ===============
    
    # ===============
    # res = self.to('cpu') * self.cmp()
    # return res
    # ===============

    # ===============
    # y = self * self.ge(0)
    # return y
    # ===============


    # x = self.to('cpu') * self.to('cpu').fss()
    return self.view(-1) * self.fss()
    


@timeit
@track_network_traffic
def nss_leakyrelu(self, alpha=0.02):
    """Compute a Leaky Rectified Linear function on the input tensor."""

    # ======================
    res = self * self.dicf()*(1-alpha) + alpha * self
    res = res.get_plain_text()
    # ======================
    return res
    # return self.view(-1).to('cpu') * self.to('cpu').fss() + alpha * self.view(-1).to('cpu') * (1 - self.to('cpu').fss())

@timeit
@track_network_traffic
def LPGAN_leakyrelu(self, alpha=0.02):
    """Compute a Leaky Rectified Linear function on the input tensor."""

    
    res = self.to('cpu') * self.cmp() *(1-alpha) + alpha *  self.to('cpu')
    res = res.get_plain_text()
    # ======================
    return res

@timeit
@track_network_traffic
def CrypTen_leakyrelu(self, alpha=0.02):
    """Compute a Leaky Rectified Linear function on the input tensor."""
    print("=== in leakyrelu:")

    res =  self * self.ge(0) + alpha * self * self.lt(0)
    res = res.get_plain_text()
    # ======================
    return res
    
# @track_network_traffic
# def relu(self):
#     """Compute a Rectified Linear function on the input tensor."""
#     # print("======in logic's relu=====")
#     res = self * self.ge(0)
#     return res

# def leaky_relu(self, alpha=0.02):
#     """Compute a Leaky Rectified Linear function on the input tensor."""
#     return self * self.ge(0) + alpha * self * self.lt(0)

# def leaky_relu(self, alpha=0.01):
#     """Compute a Leaky Rectified Linear function on the input tensor."""
#     return self * self.ge(0) + alpha * self * (1 - self.ge(0))


def hardtanh(self, min_value=-1, max_value=1):
    r"""Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
    """
    intermediate = crypten.stack([self - min_value, self - max_value]).relu()
    intermediate = intermediate[0].sub(intermediate[1])
    return intermediate.add_(min_value)


def where(self, condition, y):
    """Selects elements from self or y based on condition

    Args:
        condition (torch.bool or MPCTensor): when True yield self,
            otherwise yield y
        y (torch.tensor or MPCTensor): values selected at indices
            where condition is False.

    Returns: MPCTensor or torch.tensor
    """
    if is_tensor(condition):
        condition = condition.float()
        y_masked = y * (1 - condition)
    else:
        # encrypted tensor must be first operand
        y_masked = (1 - condition) * y

    return self * condition + y_masked
