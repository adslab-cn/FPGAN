import math
import random
import crypten
import torch
import torch.distributed as dist
from crypten import mpc
from crypten.lpgan.SecSRC import SRC
import crypten.communicator as comm 

@mpc.run_multiprocess(world_size=2)
def run_SRC():
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()

    # 创建加密张量
    u_enc = crypten.cryptensor([[4,10,9]], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor([[0,0,0]], ptype=crypten.mpc.arithmetic)
    # S1 == R0, S2 == R1
    # 1) 2)
    # 提取当前参与方的 _tensor 属性 ，即提取份额
    u_shared = u_enc._tensor.data
    v_shared = v_enc._tensor.data
    
    m, p= SRC(u_shared)
    if rank == 0:
        m1 = m
    else:
        m2 = m
    


run_SRC()