import random
import crypten
import torch
import torch.distributed as dist
from crypten import mpc
import crypten.communicator as comm 
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.lpgan.Secfunctions import SecAdd

def find_pi(u):
    """
    找到每个 u_i 的最小 p_i，使得 u_i * 2^(-p_i) 落在指定的范围内。
    
    对于 u_i > 0，目标范围是 [1/4, 1/2)
    对于 u_i < 0，目标范围是 (-1/4, -1/8)
    """
    u = u.double()
    # print("u:", u)
    # 初始化 p 张量，形状与 u 相同，初始值为 0
    p = torch.zeros_like(u, dtype=torch.int32)

    # 展平输入以便逐个元素处理
    u_flat = u.flatten()
    p_flat = p.flatten()

    for i in range(len(u_flat)):
        u_i = u_flat[i]
        if u_i > 0:
            p_i = 0
            while not (0.25 <= u_i * 2 ** -p_i < 0.5):
                p_i += 1
        elif u_i < 0:
            p_i = 0
            while not (-0.25 < u_i * 2 ** -p_i < -0.125):
                p_i += 1
        else:
            raise ValueError("u_i 不能为 0")
        p_flat[i] = p_i

    # 将平展后的 p 重塑回与 u 相同的形状
    p = p_flat.view_as(u)
    p = p.float()
    return p

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_shares(secret, seed):
    set_seed(seed)
    shape = secret.size()
    random_min = -1.0
    random_max = 1.0
    r = (random_max - random_min) * torch.rand(shape, dtype=secret.dtype, device=secret.device) + random_min
    s = secret - r
    return r, s

def PRZS(tensor, *size, device=None):
    """
    Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

    This function does so by generating `n` numbers across `n` parties with
    each number being held by exactly 2 parties. One of these parties adds
    this number while the other subtracts this number.
    """
    from crypten import generators
    # print("==========PRZS=============")
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    # g0 = generators["prev"][device]
    # g1 = generators["next"][device]
    
    rank = crypten.communicator.get().get_rank()
    world_size = crypten.communicator.get().get_world_size()
    # Generators for the current party
    g0 = torch.Generator(device=device).manual_seed(rank)
    g1 = torch.Generator(device=device).manual_seed((rank + 1) % world_size)

    current_share = generate_random_ring_element(*size, generator=g0, device=device)
    next_share = generate_random_ring_element(*size, generator=g1, device=device)
    tensor = current_share - next_share
    return tensor

# def extract(u):
#     """
#     截取张量中每个元素的末7位数字并保留符号。
#     """
#     u_last_seven = torch.zeros_like(u, dtype=torch.int64)
#     for i in range(u.size(0)):
#         for j in range(u.size(1)):
#             element = u[i, j].item()
#             sign = '-' if element < 0 else ''
#             abs_element_str = str(abs(element))
#             last_seven_digits = abs_element_str[-7:]
#             signed_last_seven_digits = int(sign + last_seven_digits)
#             u_last_seven[i, j] = signed_last_seven_digits
#     return u_last_seven
def extract(u):
    """
    截取张量中每个元素的末7位数字并保留符号，适用于任意维度的张量。(因为份额太大了)
    """
    original_shape = u.shape  # 保存原始张量形状
    flattened_u = u.view(-1)  # 将张量展平为一维

    # 创建一个同样大小的零张量，用于存储处理后的结果
    flattened_last_seven = torch.zeros_like(flattened_u, dtype=torch.int64)
    
    for i in range(flattened_u.size(0)):
        element = flattened_u[i].item()
        sign = '-' if element < 0 else ''
        abs_element_str = str(abs(element))
        last_seven_digits = abs_element_str[-7:] if len(abs_element_str) > 7 else abs_element_str
        signed_last_seven_digits = int(sign + last_seven_digits)
        flattened_last_seven[i] = signed_last_seven_digits

    # 将处理后的一维张量重新形状为原始维度
    result = flattened_last_seven.view(original_shape)
    return result

def SRC(u):
    """
    input u1, u2(也是提取份额进行操作的算法)
    output m1, m2, p
    """
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    # print(f"Rank {rank} - Starting SRC")
    # 1) a = u * 2 **(-p)
    # crypten.print(f"Rank:{rank}\nu:{u}", in_order=True)
    u = extract(u)
    # crypten.print(f"Rank{rank}\n=====末7位u=====:{u}", in_order=True)
    u = u.double()

    if rank == 0:
        
        p1 = find_pi(u)
        a1 = u * (2 ** (-p1))
        p2 = torch.zeros_like(p1)
        # print(f"Rank {rank} - p1: {p1}")
        # print(f"Rank {rank} - a1: {a1}")
    else:
        p2 = find_pi(u)
        a2 = u * (2 ** (-p2))
        p1 = torch.zeros_like(p2)
        # print(f"Rank {rank} - p2: {p2}")
        # print(f"Rank {rank} - a2: {a2}")
    
    dist.barrier()
    # print(f"Rank {rank} - After first barrier")
    dist.broadcast(p1, src=0)
    dist.broadcast(p2, src=1)
    
    # 2) split a into a', a'', in which has two values—— a1 and a2
    if rank == 0:
        # a1_1 = PRZS(a1, a1.size(), device=a1.device)
        # a1_2 = a1 - a1_1
        a1_1, a1_2 = generate_shares(a1, rank)
        # print("a1_1's type:", a1_1.size())
        a2_1 = torch.zeros_like(a1_2)
        # print(f"Rank {rank} - a1_1: {a1_1}")
        # print(f"Rank {rank} - a1_2: {a1_2}")
    else:
        # a2_1 = PRZS(a2, a2.size(), device=a2.device)
        # a2_2 = a2 - a2_1
        a2_1, a2_2 = generate_shares(a2, rank)
        # print("a2_1's type:", a2_1.size())
        a1_2 = torch.zeros_like(a2_1)
        # print(f"Rank {rank} - a2_1: {a2_1}")
        # print(f"Rank {rank} - a2_2: {a2_2}")
    
    dist.barrier()
    # print(f"Rank {rank} - After second barrier")
    dist.broadcast(a1_2, src=0)
    dist.broadcast(a2_1, src=1)
    # crypten.print(f"Rank {rank} - After broadcasting a1_2 and a2_1")
    
    # 4) S1 computes p = max(p1, p2), m1 = a'1*2^(p1-p) + a'2*2^(p2-p)
    if rank == 0:
        p = torch.max(p1, p2)
        # crypten.print(f"Rank {rank} - p: {p}")
        m1 = a1_1 * (2 ** (p1 - p)) + a2_1 * (2 ** (p2 - p))
        # crypten.print(f"Rank {rank} - m1: {m1}")
        m2 = torch.zeros_like(m1)
    else:
        p = torch.max(p1, p2)
        # crypten.print(f"Rank {rank} - p: {p}")
        m2 = a1_2 * (2 ** (p1 - p)) + a2_2 * 2 ** (p2 - p)
        # crypten.print(f"Rank {rank} - m2: {m2}")
        m1 = torch.zeros_like(m2)
    
    # crypten.print(f"p_max: {p}")
    dist.barrier()
    dist.broadcast(m1, src=0)
    dist.broadcast(m2, src=1)
    m = m1 + m2
    dist.broadcast(p, src=0)

    # m = (m1 + m2)/2**16
    
    u = SecAdd(u)/2**16
    p = -(torch.log(m/u))/torch.log(torch.tensor(2.0))
    # crypten.print("u:", u)
    # crypten.print("p:", p)
    if rank == 0:
        return m1, p
    else:
        return m2, p


