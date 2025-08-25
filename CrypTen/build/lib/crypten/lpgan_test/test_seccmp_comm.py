import torch
import sys
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.lpgan import SecCMP2
import torch.distributed as dist
from crypten.mpc.provider.ttp_provider import TTPClient
from crypten.lpgan.Secfunctions import SecMul, SecAnd
from crypten.lpgan.Secfunctions import SecAnd, SecOr, get_network_bytes, calculate_network_traffic, timeit, print_execution_times, print_communication_costs
import time

crypten.init()

import psutil

# @mpc.run_multiprocess(world_size=2)
@timeit
def run_SecCmp():
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()

    # 创建加密张量
    u_enc = crypten.cryptensor(torch.randn([256,20]), ptype=crypten.mpc.arithmetic)
    start_time = time.time()
    # u_enc = crypten.cryptensor([1], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor([0], ptype=crypten.mpc.arithmetic)

    net_io_before = get_network_bytes()
    # f = u_enc.relu()
    f = SecCMP2.SecCmp(u_enc, v_enc)
    # for i in range(1000):

    #     f = SecCMP_comm.SecCmp(u_enc, v_enc)

    # # u = u_enc * v_enc
    # u = SecAnd(input_enc._tensor.data, v_enc._tensor.data)
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    crypten.print("traffic_SecCMP:",traffic_diff) # 总通信成本
    crypten.print(f"f_cmp:{f}")
    
    runtime = time.time() - start_time # 总运行时间
    print(f"executed in {runtime:.4f} seconds")
    print_execution_times() # offline 的时间成本
    print_communication_costs() # offline 的通信成本
run_SecCmp()


# import concurrent.futures
# import random
# import time
# """
# 实验1：测试在32位上的一千次操作成本
# """
# # 定义要执行的操作
# def perform_operation(a, b, operation):
#     if operation == 'cmp':
#         # return SecCMP_comm.SecCmp(a, b)
#         return a.ge(b)
#     elif operation == 'subtract':
#         return a - b
#     elif operation == 'multiply':
#         return a * b
#     elif operation == 'divide':
#         return a / b if b != 0 else float('inf')  # 避免除以零
#     else:
#         raise ValueError("Unsupported operation")

# # 主实验函数
# def run_experiment(num_operations=1000, operation='cmp'):
#     results = []
#     # 随机生成操作数
#     for _ in range(num_operations):
#         # a = random.randint(0, 2**32 - 1)  # 生成32位随机数
#         # b = random.randint(0, 2**32 - 1)
#         a = crypten.cryptensor([1], ptype=crypten.mpc.arithmetic)
#         b = crypten.cryptensor([1], ptype=crypten.mpc.arithmetic)
#         results.append(perform_operation(a, b, operation))   
#     return results

# # 使用ThreadPoolExecutor进行并行处理
# @mpc.run_multiprocess(world_size=2)
# def main():
#     num_experiments = 1  # 你想要进行的实验次数
#     operation = 'cmp'  # 选择操作类型，例如 'add', 'subtract', 'multiply', 'divide'

#     start_time = time.time()
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(run_experiment, 1000, operation) for _ in range(num_experiments)]
        
#         # 获取结果
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 results = future.result()
#                 # 你可以在这里处理每个实验的结果
#             except Exception as e:
#                 print(f'Experiment generated an exception: {e}')
    
#     end_time = time.time()
#     print(f'Executed {num_experiments} experiments of {1000} operations in {end_time - start_time:.2f} seconds.')

# if __name__ == "__main__":
#     main()
