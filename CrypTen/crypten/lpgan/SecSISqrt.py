import math
import random
import crypten
import torch
import torch.distributed as dist
from crypten import mpc
import crypten.communicator as comm 
from crypten.lpgan.SecSRC import SRC
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from crypten.lpgan.Secfunctions import SecAdd


def SISqrt(u, I):
    """
    Input:u1, u2; Iterative number: I
    Output:f1, f2
    """
    u = u._tensor.data
    m, p = SRC(u)
    rank = comm.get().get_rank()
    if rank == 0:
        alpha = -0.8099868542
        beta = 1.787727479
        y = alpha * m + beta
        h = 1/2 * (alpha * m + beta)
        crypten.print(f"Rank:{rank}\ny1:{y}", in_order=True)
        crypten.print(f"Rank:{rank}\nh1:{h}", in_order=True)
        
    else:
        alpha = -0.8099868542
        beta = 1.787727479
        y = alpha * m
        h = 1/2 * (alpha * m)
        crypten.print(f"Rank:{rank}\ny2:{y}", in_order=True)
        crypten.print(f"Rank:{rank}\nh2:{h}", in_order=True)

    dist.barrier()
    m = SecAdd(m)
    y = SecAdd(y)
    h = SecAdd(h)
    crypten.print("========m========:", m)
    crypten.print("========y========:", y)
    crypten.print("========h========:", h)

    m = ArithmeticSharedTensor(m)
    y = ArithmeticSharedTensor(y)
    crypten.print("========m_dec========:", m.get_plain_text())
    
    # 4) SecMul(m, y0)
    g = m * y

    # 5) iterative
    i = 0
    h = ArithmeticSharedTensor(h)
    crypten.print(f"========h_dec========:{h.get_plain_text()}")

    while i < I:
        # 6) SecMul(g,h)
        r = g * h
        crypten.print(f"\n===============循环:{i}===============\nr before add:{ r.get_plain_text()}")
        crypten.print(f"g:{g.get_plain_text()}")
        crypten.print(f"h:{h.get_plain_text()}\n")

        r = r._tensor.data.double()
        # 7)
        if rank == 0:
            r = 3/2 - r/2**16
            crypten.print(f"Rank:{rank}\nr1(3/2 -r):{r}", in_order=True)
        else:
            r = -r/2**16
            crypten.print(f"Rank:{rank}\nr2(-r):{r}\n", in_order=True)
        dist.barrier()
        r = SecAdd(r)
        r = r
        crypten.print("=====r after add=====", r)
        
        r = ArithmeticSharedTensor(r)
        crypten.print(f"r变回原始数量级:{r.get_plain_text()}")
        # 8) g=SecMul(g, r), h=SecMul(h, r)
        crypten.print(f"\n=========8)i:{i}=========\ng:{g.get_plain_text()}")
        g = g * r
        h = h * r
        crypten.print(f"i:{i}\ng:{g.get_plain_text()}")
        i += 1

    crypten.print(f"\nin 8)Rank:{rank} g_dec:{g.get_plain_text()}")
    h = h._tensor.data
    # 9) a = 1 if p is even else a = 2**(1/2)
    a = torch.where(p % 2 == 0, torch.tensor(1.0), torch.tensor(math.sqrt(2)))
    # 10) p_ = p/2(向下取整)
    p_ = torch.floor(p/2)
    # 11) 12) 13)
    if rank == 0:
        crypten.print(f"Rank:{rank} h:{h}", in_order=True)
        f1 = 2/(a * 2**(p_)) * h
        return f1
    else:
        crypten.print(f"Rank:{rank} h:{h}", in_order=True)
        f2 = 2/(a * 2**(p_)) * h
        return f2

   
    