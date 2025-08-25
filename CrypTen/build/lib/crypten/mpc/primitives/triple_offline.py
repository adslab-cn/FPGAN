# offline_generate.py

import pickle
import crypten.communicator as comm
import torch
import random
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

class TripleStore:
    def __init__(self, filepath):
        self.filepath = filepath
        self.additive_triples = []
        self.squares = []
        self.binary_triples = []
        self.wraps = []
        self.b2a = []

    def generate_and_store_additive_triples(self, num_triples, size0, size1, op, device=None):
        for _ in range(num_triples):
            a = generate_random_ring_element(size0, device=device)
            b = generate_random_ring_element(size1, device=device)
            c = getattr(torch, op)(a, b)

            a = ArithmeticSharedTensor(a, precision=0, src=0)
            b = ArithmeticSharedTensor(b, precision=0, src=0)
            c = ArithmeticSharedTensor(c, precision=0, src=0)

            self.additive_triples.append((a, b, c))

    def generate_and_store_squares(self, num_squares, size, device=None):
        for _ in range(num_squares):
            r = generate_random_ring_element(size, device=device)
            r2 = r.mul(r)

            stacked = torch_stack([r, r2])
            stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
            r, r2 = stacked[0], stacked[1]

            self.squares.append((r, r2))

    def generate_and_store_binary_triples(self, num_triples, size0, size1, device=None):
        for _ in range(num_triples):
            a = generate_kbit_random_tensor(size0, device=device)
            b = generate_kbit_random_tensor(size1, device=device)
            c = a & b

            a = BinarySharedTensor(a, src=0)
            b = BinarySharedTensor(b, src=0)
            c = BinarySharedTensor(c, src=0)

            self.binary_triples.append((a, b, c))

    def generate_and_store_wraps(self, num_wraps, size, device=None):
        for _ in range(num_wraps):
            num_parties = 3  # example, should match the actual number of parties
            r = [generate_random_ring_element(size, device=device) for _ in range(num_parties)]
            theta_r = torch.tensor([count_wraps([r[i]]) for i in range(num_parties)])

            shares = torch.stack(r).tolist()  # replace with actual scatter in real environment
            r = ArithmeticSharedTensor.from_shares(shares, precision=0)
            theta_r = ArithmeticSharedTensor(theta_r, precision=0, src=0)

            self.wraps.append((r, theta_r))

    def generate_and_store_b2a(self, num_b2a, size, device=None):
        for _ in range(num_b2a):
            r = generate_kbit_random_tensor(size, bitlength=1, device=device)
            rA = ArithmeticSharedTensor(r, precision=0, src=0)
            rB = BinarySharedTensor(r, src=0)

            self.b2a.append((rA, rB))

    def save_to_file(self):
        with open(self.filepath, 'wb') as f:
            pickle.dump({
                "additive_triples": self.additive_triples,
                "squares": self.squares,
                "binary_triples": self.binary_triples,
                "wraps": self.wraps,
                "b2a": self.b2a,
            }, f)

    def load_from_file(self):
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f)
            self.additive_triples = data["additive_triples"]
            self.squares = data["squares"]
            self.binary_triples = data["binary_triples"]
            self.wraps = data["wraps"]
            self.b2a = data["b2a"]

    def get_additive_triple(self):
        if not self.additive_triples:
            raise ValueError("No more additive triples available")
        return self.additive_triples.pop()

    def get_square(self):
        if not self.squares:
            raise ValueError("No more squares available")
        return self.squares.pop()

    def get_binary_triple(self):
        if not self.binary_triples:
            raise ValueError("No more binary triples available")
        return self.binary_triples.pop()

    def get_wrap(self):
        if not self.wraps:
            raise ValueError("No more wraps available")
        return self.wraps.pop()

    def get_b2a(self):
        if not self.b2a:
            raise ValueError("No more B2A values available")
        return self.b2a.pop()

if __name__ == "__main__":
    store = TripleStore('triples.pkl')
    store.generate_and_store_additive_triples(num_triples=1000, size0=(10,), size1=(10,), op='mul')
    store.generate_and_store_squares(num_squares=1000, size=(10,))
    store.generate_and_store_binary_triples(num_triples=1000, size0=(10,), size1=(10,))
    store.generate_and_store_wraps(num_wraps=1000, size=(10,))
    store.generate_and_store_b2a(num_b2a=1000, size=(10,))
    store.save_to_file()
    print("Offline generation of triples completed and stored in 'triples.pkl'")
