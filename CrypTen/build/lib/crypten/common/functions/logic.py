#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
from crypten.common.tensor_types import is_tensor
from NssMPClib.NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPClib.NssMPC.crypto.aux_parameter import DCFKey
from NssMPClib.NssMPC import RingTensor

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
    "fastleakyrelu",
    "nss",
    "fastsecnet",
    "fss",
    "cmp",
    "leaky_relu",
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
    from crypten.lpgan.Secfunctions import SecAnd, SecOr, get_network_bytes, calculate_network_traffic
    import time
    u_shared = u_shared._tensor.data
    v_shared = crypten.cryptensor(0)
    v_shared = v_shared._tensor.data
    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()
    
    # 1)2) a=u1-v1,b=v2-u2
    if rank == 0:
        a = u_shared - v_shared
    elif rank == 1:
        b = v_shared - u_shared

    dist.barrier()
    crypten.print(f"Rank {rank} barrier 1 passed")

    net_io_before = get_network_bytes()
    if rank == 0:
        a_signed_bit, a_unsigned_bit = DFC.DFC(a, 32)
        a_bit = a_unsigned_bit
        r0, s0 = provider.generate_shares(a_bit)
        s1 = torch.zeros_like(s0)

        a_sign = DFC.extract_sign_bits(a_signed_bit)
        b_sign = torch.zeros_like(a_sign)
    elif rank == 1:
        b_signed_bit, b_unsigned_bit = DFC.DFC(b, 32)
        b_bit = b_unsigned_bit
        r1, s1 = provider.generate_shares(b_bit)
        s0 = torch.zeros_like(s1)

        b_sign = DFC.extract_sign_bits(b_signed_bit)
        a_sign = torch.zeros_like(b_sign)

    dist.barrier()
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)
    crypten.print(f"Rank {rank} barrier 2 passed")
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecXor:",traffic_diff)

    # 3) SecXor(a_bit, b_bit)
    if rank == 0:
        c_bit = r0 ^ s1
    elif rank == 1:
        c_bit = r1 ^ s0
    dist.barrier()


    # 4) SecMSB(c_bit', c_bit'')
    net_io_before = get_network_bytes()
    d_bit = SecMSB.SecMSB(c_bit)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecMSB:",traffic_diff)

    net_io_before = get_network_bytes()
    # 5) e_bit = SecAnd(a_bit, d_bit)
    # a_bit from S0, d_bit from S0 and S1
    a, b, c = provider.generate_binary_triple(d_bit.size(), d_bit.size(), device=d_bit.device)
    a = a._tensor.data
    b = b._tensor.data
    c = c._tensor.data

    # if rank == 0:
    #     e1 = r0 ^ a
    #     f1 = d_bit ^ b
    # elif rank == 1:
    #     e2 = s0 ^ a
    #     f2 = d_bit ^ b
    #     e1 = torch.zeros_like(e2)
    #     f1 = torch.zeros_like(f2)
    # dist.barrier()
    # dist.broadcast(e1, src=0)
    # dist.broadcast(f1, src=0)
    # if rank == 1:
    #     e = e1 ^ e2
    #     f = f1 ^ f2
    # elif rank == 0:
    #     e = torch.zeros_like(e1)
    #     f = torch.zeros_like(f1)
    # dist.barrier()
    # dist.broadcast(e, src=1)
    # dist.broadcast(f, src=1)

    if rank == 0:
        e = r0 ^ a
        f = d_bit ^ b
    elif rank == 1:
        e = s0 ^ a
        f = d_bit ^ b

    # Step 2: Gather e and f from both parties
    e_list = [torch.zeros_like(e) for _ in range(2)]
    f_list = [torch.zeros_like(f) for _ in range(2)]
    dist.all_gather(e_list, e)
    dist.all_gather(f_list, f)

    # Step 3: Compute e and f
    e = e_list[0] ^ e_list[1]
    f = f_list[0] ^ f_list[1]

    if rank == 0:
        z0 = c ^ a&f ^ b&e ^ e&f
        e_bit = z0
    elif rank == 1:
        z1 = c ^ a&f ^ b&e
        e_bit = z1
    dist.barrier()
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecAnd:",traffic_diff)

    net_io_before = get_network_bytes()
    # 7) SecOr(a_sign, b_sign)
    lota = SecOr(a_sign, b_sign)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecOr:",traffic_diff)

    # 6) 
    Xi = e_bit.sum(dim=-1) 
    zeta = d_bit.sum(dim=-1)

    # 8) v = SecXor(lota, Xi)
    net_io_before = get_network_bytes()
    v = lota ^ Xi # if a,b<0 Xi=0 lota=1
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecXor:",traffic_diff)

    # 9) SecAnd(v, zeta)
    net_io_before = get_network_bytes()
    f = SecAnd(v, zeta)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecAnd:",traffic_diff)
    
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

    return f.to(torch.int64)


import funshade
import numpy as np
import torch
import crypten
import crypten.communicator as comm
import torch.distributed as dist
import crypten.mpc as mpc
from crypten.lpgan.Secfunctions import track_network_traffic


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
    rank = comm.get().get_rank()
    if rank == 0:
        z_0 = z + r_in0
        z_0 = z_0.numpy().astype(np.int32)
        z_0 = torch.tensor(z_0)  # Convert to torch.Tensor
        z_1 = torch.zeros_like(z_0)  # Initialize z_1 to zeros
    else:
        z_1 = z + r_in1
        z_1 = z_1.numpy().astype(np.int32)
        z_1 = torch.tensor(z_1)  # Convert to torch.Tensor
        z_0 = torch.zeros_like(z_1)  # Initialize z_0 to zeros
    dist.barrier()
    dist.broadcast(z_0, src=0)
    dist.broadcast(z_1, src=1)
    z_0 = z_0.view(-1)
    z_1 = z_1.view(-1)

    # Compute the comparison to the threshold theta
    if rank == 0:
        o_j_1 = funshade.eval_sign(K, rank, k0, z_0.numpy(), z_1.numpy())
        o_j_1 = torch.tensor(o_j_1)  # Convert to torch.Tensor
        o_j_2 = torch.zeros_like(o_j_1)
        print("o_j_i'size:", o_j_1.size())
    else:
        o_j_2 = funshade.eval_sign(K, rank, k1, z_1.numpy(), z_0.numpy())
        o_j_2 = torch.tensor(o_j_2)  # Convert to torch.Tensor
        o_j_1 = torch.zeros_like(o_j_2)

    dist.barrier()
    dist.broadcast(o_j_1, src=0)
    dist.broadcast(o_j_2, src=1)

    o = o_j_1 + o_j_2
    print("====fss====", flush=True)
    return o

@track_network_traffic
def fastsecnet(u):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    from NssMPClib.NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
    from NssMPClib.NssMPC.crypto.aux_parameter import DCFKey
    from NssMPClib.NssMPC import RingTensor
    try:
        from NssMPClib.NssMPC import RingTensor
        print("RingTensor imported successfully:", RingTensor)
    except ImportError as e:
        print("Failed to import RingTensor:", e)

    r = 9
    # r = random.randint(0, 100)
    alpha = RingTensor.convert_to_ring(r)

    b00, b01 = generate_share(1)
    b10, b11 = generate_share(-r)

    r0, r1 = generate_share(r)
    beta = RingTensor.convert_to_ring([1,-r])
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

    Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=-beta)

    k0 = [Key0, r0, b00, b10]
    k1 = [Key1, r1, b01, b11]

    # 唯一的一次通信
    y = y.get_plain_text().int()

    x = RingTensor.convert_to_ring(y)
    rank = comm.get().get_rank()
    r_0 = DCF.eval(x=x, keys=Key0, party_id=0)
    r_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    dist.barrier()

    x = x.unsqueeze(1)  
    b0 = torch.tensor([k0[2], k0[3]]).to('cuda:0')
    b1 = torch.tensor([k1[2], k1[3]]).to('cuda:0')

    # b = recover_secret(b0,b1)

    sb0p = (r_0.tensor + b0)
    sb1p = (r_1.tensor + b1)
    if rank == 0:
        b0p = (res_0.tensor + b0)
        print("x.size:", x.size())
        print("b0p[:,0].size:", b0p[:,0].size())
        z0 = elementwise_multiply(b0p[:,0], x)
        y0 =  elementwise_add(z0, b0p[:,1])
        print("y0.size:", y0[0].size())
        y0 = y0[0]
        y1 = torch.zeros_like(y0)

    else:
        b1p = (res_1.tensor + b1)
        z1 = elementwise_multiply(b1p[:,0], x)
        y1 =  elementwise_add(z1, b1p[:,1])
        y1 = y1[0]
        y0 = torch.zeros_like(y1)

    dist.barrier()
    dist.broadcast(y0, src=0)
    dist.broadcast(y1, src=1)

    y = recover_secret(y0,y1)
    print("multi_y:", y)


@track_network_traffic
def fastleakyrelu(u, a=0.02):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    r = 9
    # r = random.randint(0, 100)
    alpha = RingTensor.convert_to_ring(r)

    b00, b01 = generate_share(1)
    b10, b11 = generate_share(-a*r)
    b20, b21 = generate_share(-r)
    print("b10:", b10)
    print("b11:", b11)

    r0, r1 = generate_share(r)
    beta = RingTensor.convert_to_ring([1,-a*r,-r])
    mask = torch.tensor([0,0,1])
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

    # 唯一的一次通信
    y = y.get_plain_text().int()

    x = RingTensor.convert_to_ring(y)
    rank = comm.get().get_rank()


    # print("res1:", res_1)
    # Party 0:
    if rank == 0:
        res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    dist.barrier()

    x = x.unsqueeze(1)  
    b0 = torch.tensor([k0[2], k0[3], k0[4]]).to('cuda:0')
    b1 = torch.tensor([k1[2], k1[3], k1[4]]).to('cuda:0')

    a = torch.tensor(a)
    scale = 100
    a = torch.tensor(a*scale).to(torch.int64)
    if rank == 0:
        b0p = res_0.tensor + b0[2]*mask
        print("b10:", b0[1])
        print("b0p[:,0]:", b0p[:,0])
        c = torch.tensor((a*x.tensor)*scale).to(torch.int64)
        print("c", c)
        # z1 = elementwise_multiply(b0p[:,0], c) # bp[0] * a * x
        z1 = b0p[:,0] * c
        print("z1:", z1)
        z2 = b0p[1] # bp[1] * a
        print("a:", a)
        print("z2:", z2)
        z3 = elementwise_multiply(b0p[0], x.tensor)
        z4 = -b0p[2]
        y0 = z1 + z2 + z3 + z4
        print("y0:", y0)
        y0 = y0[0]
        y1 = torch.zeros_like(y0)

    else:
        b1p = res_1.tensor + b1[2]*mask
        print("b11:", b1[1])
        print("b1p:", b1p)   
        c = torch.tensor((a*x.tensor)*scale).to(torch.int64)
        z1 = b1p[:,0] * c
        print("z1:", z1)
        z2 = b1p[1] # bp[1] * a
        print("a:", a)
        print("z2:", z2)
        z3 = elementwise_multiply(b1p[0], x.tensor)
        z4 = -b1p[2]
        y1 = z1 + z2 + z3 + z4
        print("y1:", y1)
        y1 = y1[0]
        y0 = torch.zeros_like(y1)

    dist.barrier()
    dist.broadcast(y0, src=0)
    dist.broadcast(y1, src=1)

    y = recover_secret(y0,y1)
    print("multi_y:", y)

@track_network_traffic
def nss(u):
    from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret
    net_io_before = get_network_bytes()
    r = 0
    # r = random.randint(0, 100)
    alpha = RingTensor.convert_to_ring(0)


    beta = RingTensor.convert_to_ring(0)
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
        res_0 = res_0.tensor.view(-1)
        res_1 = torch.zeros_like(res_0)

    # Party 1:
    if rank == 1:
        res_1 = DCF.eval(x=x, keys=Key1, party_id=1)
        res_1 = res_1.tensor.view(-1)
        res_0 = torch.zeros_like(res_1)

    dist.barrier()
    dist.broadcast(res_0, src=0)
    dist.broadcast(res_1, src=1)
    res = res_0 + res_1
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_nss:",traffic_diff)
    print("res:", res)
    return res

from crypten.lpgan.Secfunctions import timeit, track_network_traffic,get_network_bytes, calculate_network_traffic
@track_network_traffic
def relu(self):
    net_io_before = get_network_bytes()
    # y = self.view(-1) * self.fss()
    y = self.view(-1) * (self.nss()).view(-1)
    y = y.get_plain_text()
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_relu:",traffic_diff)
    print("y:", y)
    return y

    # return self.view(-1) * self.fss()
    # return self * self.cmp()

def leaky_relu(self, alpha=0.02):
    """Compute a Leaky Rectified Linear function on the input tensor."""
    # return self * self.cmp() + alpha * self * (1 - self.cmp())
    print("++++++++fss++++++++", flush=True)
    return self.view(-1) * self.fss() + alpha * self.view(-1) * (1 - self.fss())

# @track_network_traffic
# def relu(self):
#     """Compute a Rectified Linear function on the input tensor."""
#     print("======in logic's relu=====")
#     return self * self.ge(0)

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
