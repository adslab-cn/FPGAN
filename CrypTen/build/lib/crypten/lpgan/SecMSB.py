import torch
import crypten
from itertools import product
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.mpc.primitives.binary import BinarySharedTensor
from crypten.cryptensor import CrypTensor
import crypten.communicator as comm
from crypten.lpgan.Secfunctions import SecMul, SecAnd, get_network_bytes, calculate_network_traffic

def find_last_one_positions(tensor):
    # Flip the tensor along the last dimension
    flipped_tensor = torch.flip(tensor, dims=(-1,))

    # Find the first occurrence of '1' in the flipped tensor
    first_one_positions = flipped_tensor.argmax(dim=-1)

    # Find the maximum value along the last dimension to check if there are any 1s
    has_ones = tensor.max(dim=-1).values

    # Calculate last one positions
    last_one_positions = torch.where(
        has_ones.bool(),
        tensor.shape[-1] - first_one_positions - 1,
        torch.tensor(0, device=tensor.device)
    )

    return last_one_positions

def set_t(c_bit):

    # 计算最高有效位的索引 j
    max_indices = find_last_one_positions(c_bit)

    # 创建变量t
    t_shape = list(c_bit.shape)
    t_shape[-1] += 1  # 增加一个元素
    t = torch.zeros(t_shape, dtype=torch.float)  # 创建一个比c_bit多一个元素的张量，长度为32

    # 对于每个子张量，设置前j个元素为0，j+1到l的元素为1
    for index in product(*map(range, c_bit.shape[:-1])):  # 遍历前面的维度
        max_index = max_indices[index]  # 获取当前子张量的最高有效位索引
        t[index][:max_index.item()+1] = 0  # 将前j个元素设为0
        t[index][max_index.item()+1:] = 1   # 将j+1到l的元素设为1
    
    return t

def fbit(f_bit):
    # 获取 f_bit 的形状
    shape = f_bit.shape

    # 创建一个与 f_bit 具有相同形状的全零张量
    result = torch.zeros_like(f_bit)

    # 遍历每个子张量
    for idx in product(*map(range, shape[:-1])):
        # 获取当前子张量
        sub_tensor = f_bit[idx]

        # 找到第一个出现的1的位置
        first_one_idx = (sub_tensor == 1).nonzero(as_tuple=True)[0]
        if first_one_idx.numel() > 0:  # 确保有1存在
            result[idx][first_one_idx[0]] = 1

    return result

from crypten.lpgan import DFC
from crypten.mpc import MPCTensor
import torch.distributed as dist

def SecMSB(c_bit):
    """
    input: c_bit', c_bit''
    output: zeta = 0 if c_bit' == c_bit'' else zeta = 1
    """
    rank = comm.get().get_rank()
    # offline phase: DS generate t_l' + t_l'' = 1
    if rank == 0:
        t_l = torch.tensor([0])
    else:
        t_l = torch.tensor([1])

    # online phase:
    # 1) SecAnd(t_i+1, i-u_i), i = 0,...,l-1
    l = (c_bit.shape)[-1]
    t_bit = [None] * (l+1)
    
    # 扩展 t_l 使其与 c_bit 的维度一致
    t_bit[l] = t_l.expand_as(c_bit[..., 0])  # 将最高位 t_l 扩展到与 c_bit 的第一个维度相同

    net_io_before = get_network_bytes()
    for i in range(l-1, -1, -1):
        # 1 - u_i
        # u = 1 - c_bit[..., i] # 这一步的问题，1-u是对明文的取反，但这里是对每一份额都取反，所以合并后的明文结果仍旧是原值
        if rank == 0:
            u = 1 - c_bit[..., i]
        else:
            u = c_bit[..., i]
        v = t_bit[i+1]
        # 安全与运算 t_bit[i] = SecAnd(v, u)
        t_bit[i] = SecAnd(v, u) # 这一步的通信开销巨大，因为要进行 l-1 次的 SecAnd
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecMSB_SecAnd:",traffic_diff)

    # 将列表转换为 tensor
    t_bit = torch.stack(t_bit, dim=-1)

    # 不同份额的运算
    f_bit = t_bit[...,1:] - t_bit[...,:-1]
    # crypten.print(f"Rank:{rank}\nf_bit:{f_bit}", in_order=True)
    
    return f_bit

# def SecMSB(c_bit):
#     """
#     input: c_bit', c_bit''
#     output: zeta = 0 if c_bit' == c_bit'' else zeta = 1
#     """
#     rank = comm.get().get_rank()
#     # offline phase: DS generate t_l' + t_l'' = 1
#     t_l = torch.tensor([0]) if rank == 0 else torch.tensor([1])

#     # online phase:
#     l = (c_bit.shape)[-1]
    
#     # 直接创建 t_bit 张量
#     t_bit = torch.zeros((c_bit.shape[0], c_bit.shape[1], l + 1), dtype=c_bit.dtype)

#     # 扩展 t_l 使其与 c_bit 的维度一致
#     t_bit[..., l] = t_l.expand_as(c_bit[..., 0])  # 将最高位 t_l 扩展到与 c_bit 的第一个维度相同

#     # 使用矢量化操作替代循环
#     for i in range(l - 1, -1, -1):
#         # 计算 u，减少维度
#         u = c_bit[..., i] if rank else 1 - c_bit[..., i]
#         v = t_bit[..., i + 1]

#         # 可以尝试对 v 和 u 进行 squeeze 操作，以减小维度
#         # 例如：
#         v_squeezed = v.squeeze()  # 删除大小为 1 的维度
#         u_squeezed = u.squeeze()  # 删除大小为 1 的维度

#         # 计算 SecAnd，并传入减少维度后的张量
#         t_bit[..., i] = SecAnd(v_squeezed, u_squeezed)

#     # 不同份额的运算
#     f_bit = t_bit[..., 1:] - t_bit[..., :-1]
    
#     return f_bit

# def SecMSB(c_bit):
#     """
#     input: c_bit', c_bit''
#     output: zeta = 0 if c_bit' == c_bit'' else zeta = 1
#     """
#     rank = comm.get().get_rank()
#     # offline phase: DS generate t_l' + t_l'' = 1
#     if rank == 0:
#         t_l = torch.tensor([0])
#     else:
#         t_l = torch.tensor([1])

#     # online phase:
#     # 1) SecAnd(t_i+1, i-u_i), i = 0,...,l-1
#     l = (c_bit.shape)[-1]
    
#     if rank == 0:
#         u = 1 - c_bit[...,]
#     else:
#         u = c_bit[...,]
#     t_bit = SecAnd(t_l, u)

#     # 确保 t_l 的维度与 t_bit 匹配
#     t_l = t_l.unsqueeze(-1).unsqueeze(-1).expand_as(t_bit[..., -1:])

#     # 拼接 t_l 到 t_bit 的末尾
#     t_bit = torch.cat((t_bit, t_l), dim=-1)

#     # 不同份额的运算
#     f_bit = t_bit[...,1:] - t_bit[...,:-1]
#     crypten.print(f"Rank:{rank}\nf_bit:{f_bit}", in_order=True)
    
#     return f_bit