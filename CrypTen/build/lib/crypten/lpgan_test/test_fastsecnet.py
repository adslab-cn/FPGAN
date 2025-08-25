# import the libraries
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.aux_parameter import DCFKey
from NssMPC import RingTensor
import random
import torch
import crypten
import crypten.communicator as comm
import torch.distributed as dist
from crypten.lpgan.Secfunctions import generate_share, elementwise_add, elementwise_multiply, recover_secret, print_communication_costs

num_of_keys = 4  # We need a few keys for a few function values, but of course we can generate many keys in advance.

# generate keys in offline phase
# set alpha and beta
r = 9
# r = random.randint(0, 100)
alpha = RingTensor.convert_to_ring(r)
# beta = RingTensor.convert_to_ring(1) # beta能否像FastSecNet那样，是由元组组成的？
# beta = RingTensor.convert_to_ring([1,6]) # beta能否像FastSecNet那样，是由元组组成的？




crypten.init()



def run():
    # b00, b01 = generate_share(1)
    # b10, b11 = generate_share(-r)

    # r0, r1 = generate_share(r)
    # beta = RingTensor.convert_to_ring([1,-r])
    # # ======================
    # #     online phase
    # # ======================
    # # generate some values what we need to evaluate
    # # x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x = torch.tensor([-1, -2, 9, 10])

    # u = crypten.cryptensor(x)
    # v = crypten.cryptensor(r)
    # y = u + v

    # K = 1
    # for dim in u.size():
    #     K *= dim

    # Key0, Key1 = DCF.gen(num_of_keys=K, alpha=alpha, beta=-beta)

    # k0 = [Key0, r0, b00, b10]
    # k1 = [Key1, r1, b01, b11]

    # # 唯一的一次通信
    # y = y.get_plain_text().int()

    # x = RingTensor.convert_to_ring(y)
    # rank = comm.get().get_rank()
    # r_0 = DCF.eval(x=x, keys=Key0, party_id=0)
    # r_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    # # print("res1:", res_1)
    # # Party 0:
    # if rank == 0:
    #     res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

    # # Party 1:
    # if rank == 1:
    #     res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    # dist.barrier()

    # x = x.unsqueeze(1)  
    # b0 = torch.tensor([k0[2], k0[3]]).to('cuda:0')
    # b1 = torch.tensor([k1[2], k1[3]]).to('cuda:0')

    # b = recover_secret(b0,b1)

    # sb0p = (r_0.tensor + b0)
    # sb1p = (r_1.tensor + b1)
    # if rank == 0:
    #     b0p = (res_0.tensor + b0)
    #     z0 = elementwise_multiply(b0p[:,0], x)
    #     y0 =  elementwise_add(z0, b0p[:,1])
    #     y1 = torch.zeros_like(y0)

    # else:
    #     b1p = (res_1.tensor + b1)
    #     z1 = elementwise_multiply(b1p[:,0], x)
    #     y1 =  elementwise_add(z1, b1p[:,1])
    #     y0 = torch.zeros_like(y1)

    # dist.barrier()
    # dist.broadcast(y0, src=0)
    # dist.broadcast(y1, src=1)

    # y = recover_secret(y0,y1)
    # # print("multi_y:", y)

    # z0 = elementwise_multiply(sb0p[:,0], x)
    # z1 = elementwise_multiply(sb1p[:,0], x)

    # y0 =  elementwise_add(z0, sb0p[:,1])
    # y1 =  elementwise_add(z1, sb1p[:,1])

    # # restore result
    # y = recover_secret(y0,y1)
    # # print("sigle_y:", y)

    x = crypten.cryptensor([[[-1, -2, 9, 10]]])
    # x = crypten.cryptensor(torch.randn([169,169]), ptype=crypten.mpc.arithmetic)
    x.to("cuda") # 如果调用的是nss()，需要先转换到cuda。如果是fss()，则注释掉这一步
    x.fastsecnet()
    x.relu()
    # x.nss()
    # x.fastleakyrelu()
    print_communication_costs()
run()



# def generate_shares(secret, modulus=2**16):
#     """
#     Perform two-party secret sharing in the ring of size modulus (2^16).
#     The secret is split into two shares, `share1` and `share2`, within the modulus.
#     """
#     if isinstance(secret, (int, float)):
#         # 对于 int 和 float 类型，先转换为整数，并将其限制在 [0, modulus-1] 范围内
#         secret = int(secret) % modulus
        
#         # 生成两个随机值，限制在 [0, modulus-1] 范围内
#         share1 = random.randint(0, modulus - 1)
#         share2 = (secret - share1) % modulus  # 确保 share1 + share2 ≡ secret (mod modulus)
#         return share1, share2

#     elif isinstance(secret, list):
#         # 对于 list 类型，我们对其中的每个元素进行秘密分享
#         secret = [int(x) % modulus for x in secret]  # 确保每个元素都在 [0, modulus-1] 范围内
#         share1 = [random.randint(0, modulus - 1) for _ in secret]
#         share2 = [(s - sh) % modulus for s, sh in zip(secret, share1)]  # 确保 share1 + share2 ≡ secret (mod modulus)
#         return share1, share2

#     else:
#         raise ValueError(f"Unsupported type for secret sharing: {type(secret)}")
