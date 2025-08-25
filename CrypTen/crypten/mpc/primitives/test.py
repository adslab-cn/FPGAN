# test_imports.py
import crypten
from crypten.mpc.primitives.read_ds import RandomValuesLoader

crypten.init()
loader = RandomValuesLoader()
a, b, c = loader.get_additive_triple()
print("RandomValuesLoader 导入成功并初始化。")
print("a:", a.get_plain_text())
print("b:", b.get_plain_text())
print("c:", c.get_plain_text())

