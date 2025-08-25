import torch
import crypten
import crypten.mpc as mpc
import torch.distributed as dist
import crypten.communicator as comm 
from crypten.lpgan import SecMSB
from crypten.lpgan import DFC
from crypten.mpc.primitives import BinarySharedTensor

@mpc.run_multiprocess(world_size=2)
def run_SecMSB():
    # 创建加密张量
    u_enc = crypten.cryptensor([[1,1,90]], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor([[4,1,9]], ptype=crypten.mpc.arithmetic)
    # S1 == R0, S2 == R1
    # 1) 2)
    # 提取当前参与方的 _tensor 属性 ，即提取份额
    u_shared = u_enc._tensor.data
    v_shared = v_enc._tensor.data

    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()
    
    if rank == 0:
        a = u_shared - v_shared

    elif rank == 1:
        b = v_shared - u_shared

    dist.barrier()


    if rank == 0:
        a_signed_bit, a_unsigned_bit = DFC.DFC(a, 32)
        a_bit = a_unsigned_bit

        r0, s0 = provider.generate_shares(a_bit)
        s1 = torch.zeros_like(s0)

        a_sign = DFC.extract_sign_bits(a_signed_bit)

    elif rank == 1:
        b_signed_bit, b_unsigned_bit = DFC.DFC(b, 32)
        b_bit = b_unsigned_bit
        
        r1, s1 = provider.generate_shares(b_bit)
        s0 = torch.zeros_like(s1)

        b_sign = DFC.extract_sign_bits(b_signed_bit)
        a_sign = torch.zeros_like(b_sign)


    dist.barrier()
    dist.broadcast(a_sign, src=0)
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)


    if rank == 0:
        z0 = r0 ^ s1
        z1 = torch.zeros_like(z0)
    elif rank == 1:
        z1 = r1 ^ s0
        z0 = torch.zeros_like(z1)
    dist.barrier()
    # 各节点将各自的计算结果公开
    dist.broadcast(z0, src=0)
    dist.broadcast(z1, src=1)

    # SecXor(a_bit, b_bit)
    c_bit = z0 ^ z1
    crypten.print("c_bit:", c_bit)

    r, s = provider.generate_shares(c_bit)
    if rank == 0:
        c_bit = r
    else:
        c_bit = s
    dist.barrier()

    crypten.print(f"Rank:{rank}\nc_bit:{c_bit}", in_order=True)

    f = SecMSB.SecMSB(c_bit)
    crypten.print(f"Rank:{rank}\nf:{f}", in_order=True)
    
run_SecMSB()