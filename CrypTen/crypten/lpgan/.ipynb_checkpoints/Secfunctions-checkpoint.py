import math
import random
import crypten
import torch
import torch.distributed as dist
import crypten.communicator as comm 
import psutil
import time
from functools import wraps
import numpy as np

# 全局变量用于记录函数执行时间
execution_times = {}

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        
        # 记录每个函数的执行时间
        if func.__name__ in execution_times:
            execution_times[func.__name__] += runtime
        else:
            execution_times[func.__name__] = runtime
        
        # print(f"Function {func.__name__} executed in {runtime:.4f} seconds")
        return result
    return wrapper

def print_execution_times():
    """打印所有被记录的函数的执行时间"""
    total_time = 0
    print("\nExecution Time Summary:")
    for func_name, exec_time in execution_times.items():
        print(f"Total time for {func_name}: {exec_time:.4f} seconds")
        total_time += exec_time
    print(f"\nTotal execution time: {total_time:.4f} seconds")

# 全局变量用于记录通信开销
communication_costs = {}

def get_network_bytes():
    """获取所有网卡的网络流量统计信息"""
    net_io = psutil.net_io_counters(pernic=True)
    return net_io

def calculate_network_traffic(net_io_start, net_io_end):
    """计算网络流量差异"""
    traffic = {}
    
    for interface, stats_start in net_io_start.items():
        stats_end = net_io_end.get(interface)
        if stats_end:
            bytes_sent = stats_end.bytes_sent - stats_start.bytes_sent
            bytes_received = stats_end.bytes_recv - stats_start.bytes_recv
            traffic[interface] = {
                'bytes_sent': bytes_sent,
                'bytes_received': bytes_received
            }
    
    # 解析字典并计算总的通信量（所有接口发送和接收字节总和）
    total_traffic = sum(
        t['bytes_sent'] + t['bytes_received'] for t in traffic.values()
    )
    
    return total_traffic / 1024 / 1024  # MB为单位

def track_network_traffic(func):
    @wraps(func)  # 使用wraps保留函数的元数据
    def wrapper(*args, **kwargs):
        # 获取函数调用前的网络流量信息
        net_io_start = get_network_bytes()
        
        # 调用被装饰的函数
        result = func(*args, **kwargs)
        
        # 获取函数调用后的网络流量信息
        net_io_end = get_network_bytes()
        
        # 计算网络流量差异
        traffic_diff = calculate_network_traffic(net_io_start, net_io_end)
        
        # 累加到函数的总通信开销
        if func.__name__ in communication_costs:
            communication_costs[func.__name__] += traffic_diff
        else:
            communication_costs[func.__name__] = traffic_diff
        
        # # 自动打印该函数的通信开销
        # print(f"Function '{func.__name__}' communication overhead: {traffic_diff:.4f} MB")
        
        return result
    
    return wrapper

def print_communication_costs():
    """打印所有有通信开销的函数的通信开销"""
    total_cost = 0
    print("\nCommunication Cost Summary:")
    for func_name, comm_cost in communication_costs.items():
        print(f"Total communication cost for {func_name}: {comm_cost:.4f} MB")
        total_cost += comm_cost
    print(f"\nTotal communication cost: {total_cost:.4f} MB")

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

def SecAdds(x):
    """
    利用send, recv来实现
    """
    rank = comm.get().get_rank()
    if rank == 0:
        r1, s1 = generate_shares(x, rank)
        s2 = torch.zeros_like(s1)
        comm.get().send(s1, dst=1)
        comm.get().recv(s2, src=1)
        crypten.print(f"Rank{rank}\ns1:{s1}")
    else:
        r2, s2 = generate_shares(x, rank)
        s1 = torch.zeros_like(s2)
        comm.get().send(s2, dst=0)
        comm.get().recv(s1, src=0)
        crypten.print(f"Rank{rank}\ns2:{s2}")
    if rank == 0:
        z1 = r1 + s2
        z2 = torch.zeros_like(z1)
        comm.get().send(z1, dst=1)
        comm.get().recv(z2, src=1)
    else:
        z2 = r2 + s1
        z1 = torch.zeros_like(z2)
        comm.get().send(z2, dst=0)
        comm.get().recv(z1, src=0)
    z = z1 + z2
    return z

def SecAdd(x):
    """
    S0, S1分别拥有一份x的share，要将这两个share在不被另一方知晓的前提下，
    进行安全加法
    """
    rank = comm.get().get_rank()
    if rank == 0:
        r1, s1 = generate_shares(x, rank)
        s2 = torch.zeros_like(s1)
    else:
        r2, s2 = generate_shares(x, rank)
        s1 = torch.zeros_like(s2)

    dist.barrier()
    # crypten.print(f"Rank {rank} before barrier 1", in_order=True)
    dist.broadcast(s1, src=0)
    dist.broadcast(s2, src=1)
    if rank == 0:
        z1 = r1 + s2
        z2 = torch.zeros_like(z1)
    else:
        z2 = r2 + s1
        z1 = torch.zeros_like(z2)
    dist.barrier()
    # crypten.print(f"Rank {rank} before barrier 2", in_order=True)

    dist.broadcast(z1, src=0)
    dist.broadcast(z2, src=1)
    z = z1 + z2
    return z

# def SecXor(x, y):
#     provider = crypten.mpc.get_default_provider()
#     rank = comm.get().get_rank()
#     # 基于 beaver三元组，在不暴露a_bit的情况下，计算SecAnd
#     a, b, c = provider.generate_binary_triple(d_bit.size(), d_bit.size(), device=d_bit.device)
#     a = a._tensor.data
#     b = b._tensor.data
#     c = c._tensor.data
#     # 需要分发 a, b, c
#     if rank == 0:
#         # r = generate_kbit_random_tensor(a_bit.size(), device=a_bit.device)
#         # s = generate_kbit_random_tensor(a_bit.size(), device=a_bit.device)
#         r,s = provider.generate_shares(a_bit)
#     elif rank == 1:
#         s = torch.zeros_like(d_bit)
#     dist.barrier()
#     dist.broadcast(s, src=0)

#     if rank == 0:
#         e1 = r ^ a
#         f1 = d_bit ^ b
#     elif rank == 1:
#         e2 = s ^ a
#         f2 = d_bit ^ b
#         e1 = torch.zeros_like(e2)
#         f1 = torch.zeros_like(f2)
#     dist.barrier()
#     dist.broadcast(e1, src=0)
#     dist.broadcast(f1, src=0)
#     if rank == 1:
#         e = e1 ^ e2
#         f = f1 ^ f2
#     elif rank == 0:
#         e = torch.zeros_like(e1)
#         f = torch.zeros_like(f1)
#     dist.barrier()
#     dist.broadcast(e, src=1)
#     dist.broadcast(f, src=1)
#     if rank == 0:
#         # z0 = c1+a1f+b1e+ef
#         z0 = c ^ a&f ^ b&e ^ e&f
#         z1 = torch.zeros_like(z0)
#     elif rank == 1:
#         z1 = c ^ a&f ^ b&e
#         z0 = torch.zeros_like(z1)
#     dist.barrier()
#     dist.broadcast(z0, src=0)
#     dist.broadcast(z1, src=1)
#     e_bit = z0 ^ z1

def SecMul(x, y):
    provider = crypten.mpc.get_default_provider()
    rank = comm.get().get_rank()
    a, b, c = provider.generate_additive_triple(x.size(), y.size(), op='mul', device=x.device)
    a = a._tensor.data
    b = b._tensor.data
    c = c._tensor.data
    # x = x._tensor.data
    # y = y._tensor.data

    if rank == 0:
        e1 = x - a
        f1 = y - b
    elif rank == 1:
        e2 = x - a
        f2 = y - b
        e1 = torch.zeros_like(e2)
        f1 = torch.zeros_like(f2)
    dist.barrier()
    dist.broadcast(e1, src=0)
    dist.broadcast(f1, src=0)
    if rank == 1:
        e = e1 + e2
        f = f1 + f2
    elif rank == 0:
        e = torch.zeros_like(e1)
        f = torch.zeros_like(f1)
    dist.barrier()
    dist.broadcast(e, src=1)
    dist.broadcast(f, src=1)
    if rank == 0:
        # z0 = c1+a1f+b1e+ef
        z0 = c + a*f + b*e + e*f
        z1 = torch.zeros_like(z0)
    elif rank == 1:
        z1 = c + a*f + b*e
        z0 = torch.zeros_like(z1)
    dist.barrier()
    dist.broadcast(z0, src=0)
    dist.broadcast(z1, src=1)
    result = z0 + z1

    return result

def pack_bits(tensor):
    """将布尔张量打包为 uint8 张量（每个字节存储8个布尔值）"""
    tensor_np = tensor.cpu().numpy().astype(np.uint8)  # 确保输入为 uint8
    packed = np.packbits(tensor_np, axis=None)
    return torch.tensor(packed, dtype=torch.uint8)  # 强制指定 dtype

def unpack_bits(packed_tensor, original_shape):
    # 确保输入为 uint8 类型
    if packed_tensor.dtype != torch.uint8:
        packed_tensor = packed_tensor.to(torch.uint8)
    packed_np = packed_tensor.cpu().numpy()
    unpacked_np = np.unpackbits(packed_np, axis=None, count=np.prod(original_shape))
    return torch.from_numpy(unpacked_np).reshape(original_shape).to(torch.uint8)
    
# @track_network_traffic
def SecAnd(x, y):
    net_io_before = get_network_bytes()
    provider = crypten.mpc.get_default_provider()
    rank = comm.get().get_rank()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)
    a = a._tensor.data.bool()
    b = b._tensor.data.bool()
    c = c._tensor.data.bool()
    # crypten.print(f"a:{a.size()}", in_order=True)
    
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)

    #============ 一轮通信 =============#
    # Step 1: Compute local shares of e and f
    e_unpack = x ^ a
    f_unpack = y ^ b
    # crypten.print(f"x's size:{x.size()}", in_order=True)
    # crypten.print(f"e_local's size:{e_local.size()}", in_order=True)
    # crypten.print(f"e_local's type:{e_local.dtype}", in_order=True)
    
    e_local = pack_bits(e_unpack)
    f_local = pack_bits(f_unpack)
    
    # Step 2: Gather e and f from both parties
    e_list = [torch.zeros_like(e_local) for _ in range(2)]
    f_list = [torch.zeros_like(f_local) for _ in range(2)]
    dist.all_gather(e_list, e_local)
    dist.all_gather(f_list, f_local)

    # Step 3: Compute e and f
    e = e_list[0] ^ e_list[1]
    f = f_list[0] ^ f_list[1]
    # crypten.print(f"e:{e.dtype}\nf:{f.dtype}", in_order=True)

    e = unpack_bits(e,e_unpack.shape)
    f = unpack_bits(f,f_unpack.shape)

    if rank == 0:
        # z0 = c1+a1f+b1e+ef
        z0 = c ^ a&f ^ b&e ^ e&f
        result = z0
        # crypten.print(f"result's type:{result.dtype}", in_order=True)

    elif rank == 1:
        z1 = c ^ a&f ^ b&e
        result = z1
        # crypten.print(f"result:{result}", in_order=True)

    dist.barrier()

    return result


def SecOr(a, b):
    """
    compute shares a(from S0) and b(from  S1)

    Input:  x=x1^x2, y=y1^y2
    Output: f=f1^f2
    
    SecOr:  f=x V y=x+y-xy
    algrithm: S1,S2 firstly compute m=-xy by invoking SecAnd,
              Then, Si output f1=x1+y1+m1 for i=[1,2]

    
    """
    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()

    if rank == 0:
        r0, s0 = provider.generate_shares(a)
        x = r0
        s1 = torch.zeros_like(s0)
        y = s1
    else:
        r1, s1 = provider.generate_shares(b)
        y = r1
        s0 = torch.zeros_like(s1)
        x = s0
    dist.barrier()
    dist.broadcast(s0, src=0)
    dist.broadcast(s1, src=1)
    m = SecAnd(x,y)

    f_xy = x ^ y # f_xy1 = x1 ^ y1, f_xy2 = x2 ^ y2
    # f = f_xy ^ m
    f = f_xy ^ m

    return f


def generate_share(secret, modulus=2**16-1):
    """
    Perform two-party secret sharing in the ring of size modulus (2^16).
    The secret is split into two shares, `share1` and `share2`, within the modulus.

    :param secret: The secret to be shared.
    :type secret: int or float
    :param modulus: The modulus for the ring (default is 2^16).
    :type modulus: int
    :param seed: The random seed for reproducibility (default is None).
    :type seed: int
    :return: A tuple of two shares (share1, share2).
    :rtype: tuple
    """
    # 这里先固定种子，否则不同节点会采用不同的种子生成随机数。要么我就先将beta离线处理并存储好
    seed = 42
    if seed is not None:
        random.seed(seed)

    if isinstance(secret, (int, float)):
        # 对于 int 和 float 类型，先转换为整数，并将其限制在 [0, modulus-1] 范围内
        secret = int(secret) % modulus
        
        # 生成两个随机值，限制在 [0, modulus-1] 范围内
        share1 = random.randint(0, modulus - 1)
        share2 = (secret - share1) % modulus  # 确保 share1 + share2 ≡ secret (mod modulus)
        return share1, share2
    else:
        raise TypeError("secret must be an int or float")
    
MODULUS = 2**16-1
def recover_secret(share1, share2):
    """
    Recover the secret from two shares in the ring of size 2^16.
    """
    # 对share1和share2逐元素相加并进行模运算
    secret = (share1 + share2) % MODULUS
    
    # 处理负数情况
    secret = torch.where(secret > MODULUS // 2, secret - MODULUS, secret)
    
    return secret

# def elementwise_multiply(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 检查 tensor1 和 tensor2 是否形状不匹配
#     if tensor1.shape != tensor2.shape:
#         # 如果形状不一致，通过 unsqueeze 来添加必要的维度
#         if tensor1.ndimension() == 1 and tensor2.ndimension() == 2:
#             tensor1 = tensor1.unsqueeze(1)  # 转换为 (10, 1)
#         elif tensor2.ndimension() == 1 and tensor1.ndimension() == 2:
#             tensor2 = tensor2.unsqueeze(1)  # 转换为 (10, 1)
#     # 返回逐元素相乘的结果
#     return tensor1 * tensor2


# def elementwise_add(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 如果 tensor1 和 tensor2 的维度不同，手动调整它们的形状
#     if tensor1.shape != tensor2.shape:
#         if tensor1.ndimension() == 1 and tensor2.ndimension() == 2:
#             tensor1 = tensor1.unsqueeze(1)  # 转换为 (10, 1)
#         elif tensor2.ndimension() == 1 and tensor1.ndimension() == 2:
#             tensor2 = tensor2.unsqueeze(1)  # 转换为 (10, 1)

#     # 确保两者形状一致，进行逐元素加法
#     return tensor1 + tensor2


# import torch

# def elementwise_multiply(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 使用广播机制自动对齐张量的维度
#     # 广播规则：如果两个张量的形状不匹配，PyTorch 会自动扩展较小的维度
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())
#     return tensor1 * tensor2

# def elementwise_add(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 使用广播机制自动对齐张量的维度
#     # 广播规则：如果两个张量的形状不匹配，PyTorch 会自动扩展较小的维度
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())
#     return tensor1 + tensor2

# def elementwise_multiply(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 输出张量的形状，以便调试
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())

#     # 处理 tensor2，确保它是二维的（28, 28）
#     if tensor2.ndimension() == 1:  # 如果 tensor2 是一维的
#         tensor2 = tensor2.view(28, 28)  # 将 tensor2 转换为 [28, 28]
#     elif tensor2.ndimension() == 3 and tensor2.size(1) == 1:  # 如果 tensor2 是 [28, 1, 28]
#         tensor2 = tensor2.squeeze(1)  # 去掉中间的维度，将其转换为 [28, 28]
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())
#     tensor1 = tensor1.view(28, 28)  # 或者 tensor1.reshape(28, 28)
#     # 返回逐元素相乘的结果
#     return tensor1 * tensor2


# def elementwise_add(tensor1, tensor2):
#     # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
#     if hasattr(tensor1, 'tensor'):
#         tensor1 = tensor1.tensor
#     if hasattr(tensor2, 'tensor'):
#         tensor2 = tensor2.tensor

#     # 输出张量的形状，以便调试
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())

#     # 处理 tensor2，确保它是二维的（28, 28）
#     if tensor2.ndimension() == 1:  # 如果 tensor2 是一维的
#         tensor2 = tensor2.view(28, 28)  # 将 tensor2 转换为 [28, 28]
#     elif tensor2.ndimension() == 3 and tensor2.size(1) == 1:  # 如果 tensor2 是 [28, 1, 28]
#         tensor2 = tensor2.squeeze(1)  # 去掉中间的维度，将其转换为 [28, 28]
#     print("tensor1'size:", tensor1.size())
#     print("tensor2'size:", tensor2.size())
#     # 返回逐元素加法的结果
#     return tensor1 + tensor2

def elementwise_multiply(tensor1, tensor2):
    # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
    if hasattr(tensor1, 'tensor'):
        tensor1 = tensor1.tensor
    if hasattr(tensor2, 'tensor'):
        tensor2 = tensor2.tensor

    # 获取 tensor2 的形状
    H, W = tensor2.shape[-2], tensor2.shape[-1]  # 获取tensor2的形状中的 H 和 W

    # 输出张量的形状，以便调试


    # 如果 tensor1 是一维的，则调整形状
    if tensor1.ndimension() == 1:
        N = tensor1.size(0)  # 获取 tensor1 的大小
        if N % W == 0:  # 确保 N 可以被 W 整除
            tensor1 = tensor1.view(N // W, W)  # 将 tensor1 变形为 [N / W, W]
        else:
            raise ValueError(f"tensor1's size {N} cannot be divided by tensor2's last dimension {W}")

    # 返回逐元素相乘的结果
    return tensor1 * tensor2

def elementwise_add(tensor1, tensor2):
    # 提取 RingTensor 对象的实际 tensor（如果是 RingTensor）
    if hasattr(tensor1, 'tensor'):
        tensor1 = tensor1.tensor
    if hasattr(tensor2, 'tensor'):
        tensor2 = tensor2.tensor

    # 输出张量的形状，以便调试


    # 获取 tensor1 的最后一维
    tensor1_last_dim = tensor1.shape[-1]
    
    # 如果 tensor2 是一维的，将它转换为 [64, 64, 32] 形状（根据 tensor1 的最后一维）
    if tensor2.ndimension() == 1:
        N = tensor2.size(0)  # 获取 tensor2 的大小
        if N % tensor1_last_dim == 0:  # 确保 N 可以被 tensor1 的最后一维大小整除
            tensor2 = tensor2.view(-1, tensor1_last_dim)  # 将 tensor2 变形为适当的形状
        else:
            raise ValueError(f"tensor2's size {N} cannot be divided by tensor1's last dimension {tensor1_last_dim}")
    
    # 输出转换后的 tensor2 的形状


    # 返回逐元素加法的结果
    return tensor1 + tensor2


