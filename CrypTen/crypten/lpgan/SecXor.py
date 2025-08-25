import torch
import crypten
import random
import crypten.mpc as mpc
import torch.distributed as dist
import crypten.communicator as comm 
from crypten.mpc.primitives import BinarySharedTensor
from crypten.lpgan import DFC
from crypten.lpgan import SecMSB
from crypten.common.rng import generate_random_ring_element
from crypten import generators


def SecXor(u, v):
    """compute SecXor(a, b)
    firstly: randomly geneate c(i.e r), and send c0 to S0, c1 to S1
    secondly: In S0, a is split into c0 and a-c0, S1 as the same
    thirdly: Xor = a + b - 2*a*b, 
    """

    rank = comm.get().get_rank()
    if rank == 0:
        a = u - v
        r0, s0 = generate_shares(a)
        s1 = torch.zeros_like(s0)
    else:
        b = v - u
        r1, s1 = generate_shares(b)
        s0 = torch.zeros_like(s1)

    dist.barrier()
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)
    # 打印结果
    crypten.print(f"Rank{rank}-s0 ={s0}", in_order=True)
    crypten.print(f"Rank{rank}-s1 ={s1}", in_order=True)

    if rank == 0:
        z0 = r0 ^ s1
        z1 = torch.zeros_like(z0)
    else:
        z1 = r1 ^ s0
        z0 = torch.zeros_like(z1)
    dist.barrier()
    dist.broadcast(z0, src=0)
    dist.broadcast(z1, src=1)
    crypten.print(f"Rank{rank}-z0 ={z0}", in_order=True)
    crypten.print(f"Rank{rank}-z1 ={z1}", in_order=True)
    # SecXor
    z = z0 ^ z1
    crypten.print("z:", z)


def generate_shares(secret):
    cpu_device = torch.device("cpu")
    generators["local"][cpu_device] = torch.default_generator
    r = random.randint(0, 1)
    s = secret ^ r
    return r, s

def add_shares(r0, s1, r1, s0):
    z0 = r0 ^ s1
    z1 = r1 ^ s0
    return z0, z1

def reconstruct_secret(z0, z1):
    return z0 ^ z1

# Secret values
x = 42  # Node 0
y = 23  # Node 1

# Node 0 generates shares for x
r0, s0 = generate_shares(x)
# Node 1 generates shares for y
r1, s1 = generate_shares(y)

# Exchange shares
# Node 0 now holds r0 and s1
# Node 1 now holds r1 and s0

# Each node calculates partial result
z0, z1 = add_shares(r0, s1, r1, s0)

# Combine partial results to reconstruct final result
z = reconstruct_secret(z0, z1)

print(f"Original x: {x}, y: {y}")
print(f"Shares of x: (r0: {r0}, s0: {s0})")
print(f"Shares of y: (r1: {r1}, s1: {s1})")
print(f"Partial results: (z0: {z0}, z1: {z1})")
print(f"Reconstructed z: {z}")
