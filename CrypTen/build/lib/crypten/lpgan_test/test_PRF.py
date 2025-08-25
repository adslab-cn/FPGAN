import torch
from crypten.common.rng import generate_random_ring_element
from crypten import generators

"""生成一个大小为 (3, 3) 的随机数张量"""

cpu_device = torch.device("cpu")
generators["local"][cpu_device] = torch.default_generator

size = (3, 3)
rand_element_0 = generate_random_ring_element(size)
rand_element_1 = generate_random_ring_element(size)

print(rand_element_0)
print(rand_element_1)
