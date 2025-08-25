import torch
import crypten
import crypten.mpc as mpc
import torch.distributed as dist
import crypten.communicator as comm 
from crypten.mpc.primitives import BinarySharedTensor
from crypten.lpgan import DFC
from crypten.lpgan import SecMSB

def SecCmp(u_shared, v_shared):
    u_shared = u_shared._tensor.data
    v_shared = v_shared._tensor.data
    rank = comm.get().get_rank()

    if rank == 0:
        a = u_shared - v_shared
    else:
        b = v_shared - u_shared

    # 使用 barrier 进行同步
    dist.barrier()

    # 使用广播确保两个节点都能收到结果
    if rank == 0:
        a_tensor = a
        b_tensor = torch.zeros_like(a_tensor)
    else:
        a_tensor = torch.zeros_like(u_shared)
        b_tensor = b
    
    dist.broadcast(a_tensor, src=0)
    dist.broadcast(b_tensor, src=1)

    # DFC：将 int 的绝对值转换为数值位，保留符号位
    l = 32
    crypten.print("a_tensor:", a_tensor)
    crypten.print("b_tensor:", b_tensor)

    a_signed_bit, a_unsigned_bit = DFC.DFC(a_tensor, l)
    b_signed_bit, b_unsigned_bit = DFC.DFC(b_tensor, l)
    a_bit = a_unsigned_bit
    b_bit = b_unsigned_bit
    crypten.print("a_bit:", a_bit)
    crypten.print("b_bit:", b_bit)


    # 3) SecXor(a_bit, b_bit)
    a_bit = BinarySharedTensor(a_bit)
    b_bit = BinarySharedTensor(b_bit)

    c_bit = a_bit ^ b_bit

    # 4) SecMSB(c_bit_0, c_bit_1)
    d_bit = SecMSB.SecMSB(c_bit)
    zeta = d_bit.sum(dim=-1)
    crypten.print("zeta:", zeta)

    d_bit = BinarySharedTensor(d_bit)

    
    # 5) SecAnd(a_bit, d_bit) 按理来说，这个d_bit的1的位置虽然不是固定的，但是会和大的那个值的最高1相对应
    e_bit = a_bit & d_bit

    # 计算最高有效位的索引 j
    e_bit_dec = e_bit.get_plain_text()
    max_indices = e_bit_dec.argmax(dim=-1)

    # # 6) 逐元素相加 ξ = ∑ e_i
    # #    ξ = 1 if a > b else ξ = 0
    # Xi 会有三种情况，一种是 a, b 全< 0; 一种是 a, b 全> 0；一种是 a ,b中既有正又有负
    Xi = e_bit_dec.sum(dim=-1)
    crypten.print("Xi:", Xi)

    # # 7) SecOr(a_l-1, b_l-1)符号位的比较
    # a_l = extract_sign_bits(a_signed_bit)
    # b_l = extract_sign_bits(b_signed_bit)
    # Iota = a_l or b_l

    # # 8) SecXor
    # a和 b的符号位对 Xi 进行异或——》修正负数的绝对值会带来相反结果的情况
    a_sign = DFC.extract_sign_bits(a_signed_bit)
    b_sign = DFC.extract_sign_bits(b_signed_bit)

    v = a_sign ^ Xi
    crypten.print("v:", v)

    # # 9) SecAnd 当 a = b时，e = 0，与上zeta——》修正当上一步修正符号位后，存在的 a, b<0时，符号位为 1, 但同时 a=b，Xi = 0, 于是异或结果为 1的错误情况
    f = v * zeta
    
    return f
