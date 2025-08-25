# import pickle
# import random
# import crypten
# from CrypTen.crypten.mpc.primitives.read_ds import RandomValuesLoader

# with open("random_values.pkl", "rb") as f:
#     random_values = pickle.load(f)

# # 验证每个列表中的值是否唯一
# for key, values in random_values.items():
#     unique_values = set([pickle.dumps(v) for v in values])
#     print(f"{key}: {len(unique_values) == len(values)}")


# crypten.init()
# loader = RandomValuesLoader()
# a, b, c = loader.get_additive_triple()
# print("RandomValuesLoader 导入成功并初始化。")
# print("a:", a.get_plain_text())
# print("b:", b.get_plain_text())
# print("c:", c.get_plain_text())