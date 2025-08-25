# offline_generate.py
import os
import crypten
import torch
import pickle
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

class TrustedFirstParty:
    NAME = "TFP"

    def __init__(self, storage_path="random_values.pkl", num_samples=100, size=(2, 2)):
        self.storage_path = storage_path
        self.num_samples = num_samples
        self.size = size
        self.random_values = {
            "additive_triples": [],
            "squares": [],
            "binary_triples": [],
            "wraps": [],
            "b2a": [],
            "shares": []
        }

    def save_random_values(self):
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.random_values, f)

    def generate_additive_triple(self, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        for _ in range(self.num_samples):
            a = generate_random_ring_element(self.size, device=device)
            b = generate_random_ring_element(self.size, device=device)
            c = getattr(torch, op)(a, b, *args, **kwargs)

            a = ArithmeticSharedTensor(a, precision=0, src=0)
            b = ArithmeticSharedTensor(b, precision=0, src=0)
            c = ArithmeticSharedTensor(c, precision=0, src=0)

            self.random_values["additive_triples"].append((a, b, c))

    def square(self, device=None):
        """Generate square double of given size"""
        for _ in range(self.num_samples):
            r = generate_random_ring_element(self.size, device=device)
            r2 = r.mul(r)

            # Stack to vectorize scatter function
            stacked = torch_stack([r, r2])
            stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
            
            self.random_values["squares"].append((stacked[0], stacked[1]))

    def generate_binary_triple(self, device=None):
        """Generate xor triples of given size"""
        for _ in range(self.num_samples):
            a = generate_kbit_random_tensor(self.size, device=device)
            b = generate_kbit_random_tensor(self.size, device=device)
            c = a & b

            a = BinarySharedTensor(a, src=0)
            b = BinarySharedTensor(b, src=0)
            c = BinarySharedTensor(c, src=0)

            self.random_values["binary_triples"].append((a, b, c))

    def B2A_rng(self, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        for _ in range(self.num_samples):
            r = generate_kbit_random_tensor(self.size, bitlength=1, device=device)

            rA = ArithmeticSharedTensor(r, precision=0, src=0)
            rB = BinarySharedTensor(r, src=0)

            self.random_values["b2a"].append((rA, rB))

# 示例用法
if __name__ == "__main__":
    # Initialize crypten
    crypten.init()

    # 使用当前时间设置种子
    random_seed = int.from_bytes(os.urandom(4), byteorder="little")
    torch.manual_seed(random_seed)

    # 设置Crypten的本地生成器
    crypten.generators["local"] = {
        torch.device('cpu'): torch.Generator(device='cpu').manual_seed(random_seed),
    }
    
    tfp = TrustedFirstParty(num_samples=1000, size=(10, 10))
    tfp.generate_additive_triple('mul', device='cpu')
    tfp.square()
    tfp.generate_binary_triple()
    tfp.B2A_rng()
    
    # 保存所有生成的随机值
    tfp.save_random_values()
