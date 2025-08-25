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

# @mpc.run_multiprocess(world_size=2)
def SecCmp(u, v):
    u_shared = u._tensor.data
    v_shared = v._tensor.data
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
        crypten.print(f"a_bit:{a_bit}")
        r0, s0 = provider.generate_shares(a_bit)
        # comm.get().send(s0, 1)
        s1 = torch.zeros_like(s0)
        a_sign = DFC.extract_sign_bits(a_signed_bit)
        b_sign = torch.zeros_like(a_sign)
    elif rank == 1:
        b_signed_bit, b_unsigned_bit = DFC.DFC(b, 32)
        b_bit = b_unsigned_bit
        r1, s1 = provider.generate_shares(b_bit)
        s0 = torch.zeros_like(s1)
        # s0 = comm.get().recv(s1, 0)
        b_sign = DFC.extract_sign_bits(b_signed_bit)
        a_sign = torch.zeros_like(b_sign)

# 用send/recv替换broadcast后的通信量几乎没有变化
    # if rank == 1:
    #     comm.get().send(s1, 0)
    # else:
    #     s1 = comm.get().recv(s1, 1)
    dist.barrier()
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)
    crypten.print(f"Rank {rank} barrier 2 passed")
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecXor1:",traffic_diff)

# ================================ 
#   测试crypten自带安全And的通信开销
# ================================ 
    if rank == 0:
        a = a_unsigned_bit
        b = torch.zeros_like(a_unsigned_bit)
    else:
        a = torch.zeros_like(b_unsigned_bit)
        b = b_unsigned_bit
    a = BinarySharedTensor(a)
    b = BinarySharedTensor(b)
    net_io_before = get_network_bytes()
    x = a & b
    # x_dec = x.get_plain_text()
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_Crypten:",traffic_diff)
    # 结果自带的安全与操作通信开销更大
#=====================================================#

    net_io_before = get_network_bytes()
    # 3) SecXor(a_bit, b_bit)
    if rank == 0:
        c_bit = r0 ^ s1
    elif rank == 1:
        c_bit = r1 ^ s0
    dist.barrier()
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecXor:",traffic_diff)

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

# ================================
#       方式1：send/recv
# ================================
    if rank == 0:
        e1 = r0 ^ a
        comm.get().send(e1, 1)
        f1 = d_bit ^ b
        comm.get().send(f1, 1)
    elif rank == 1:
        e2 = s0 ^ a
        f2 = d_bit ^ b
        e1 = torch.zeros_like(e2)
        f1 = torch.zeros_like(f2)
        e1 = comm.get().recv(e1, 0)
        f1 = comm.get().recv(f1, 0)

    if rank == 1:
        e = e1 ^ e2
        comm.get().send(e, 0)
        f = f1 ^ f2
        comm.get().send(f, 0)

    elif rank == 0:
        e = torch.zeros_like(e1)
        f = torch.zeros_like(f1)
        e = comm.get().recv(e, 1)
        f = comm.get().recv(f, 1)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecAnd1:",traffic_diff)
# ================================
#       方式2：两轮broadcast
# ================================
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
# ================================
#       方式3：一轮broadcast
# ================================
    # if rank == 0:
    #     e = r0 ^ a
    #     f = d_bit ^ b
    # elif rank == 1:
    #     e = s0 ^ a
    #     f = d_bit ^ b

    # # Step 2: Gather e and f from both parties
    # e_list = [torch.zeros_like(e) for _ in range(2)]
    # f_list = [torch.zeros_like(f) for _ in range(2)]
    # dist.all_gather(e_list, e)
    # dist.all_gather(f_list, f)

    # # Step 3: Compute e and f
    # e = e_list[0] ^ e_list[1]
    # f = f_list[0] ^ f_list[1]
#=========================================
    net_io_before = get_network_bytes()
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
    # print(f"f{f}")
    return f.to(torch.int64)
