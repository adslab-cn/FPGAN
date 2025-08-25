import math
import random
import crypten
import torch
import torch.distributed as dist
from crypten import mpc
import crypten.communicator as comm 
from crypten.lpgan.SecSRC import SRC
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from crypten.lpgan.SecSISqrt import SISqrt
from crypten.lpgan.Secfunctions import SecAdd


@mpc.run_multiprocess(world_size=2)
def run_SISqrt():
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()

    # 创建加密张量
    u_enc = crypten.cryptensor([[4,10,9]], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor([[0,0,0]], ptype=crypten.mpc.arithmetic)
    # S1 == R0, S2 == R1
    # 1) 2)
    # 提取当前参与方的 _tensor 属性 ，即提取份额
    # u_shared = u_enc._tensor.data
    u_shared = u_enc

    f = SISqrt(u_shared, I=9)
    f = SecAdd(f)/2**16
    crypten.print(f"f:{f}")

run_SISqrt()