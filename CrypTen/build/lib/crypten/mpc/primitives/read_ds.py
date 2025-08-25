import pickle
import random


class RandomValuesLoader:
    def __init__(self, storage_path="/home/sue/Project/PR-crypten/CrypTen/crypten/mpc/primitives/random_values.pkl"):
        self.storage_path = storage_path
        self.random_values = self.load_random_values()

    def load_random_values(self):
        with open(self.storage_path, 'rb') as f:
            return pickle.load(f)

    def get_additive_triple(self):
        return random.choice(self.random_values["additive_triples"])

    def get_square(self):
        return random.choice(self.random_values["squares"])

    def get_binary_triple(self):
        return random.choice(self.random_values["binary_triples"])

    def get_b2a(self):
        return random.choice(self.random_values["b2a"])


