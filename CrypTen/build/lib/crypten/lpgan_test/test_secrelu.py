import torch
import crypten
import crypten.mpc as mpc
from crypten.lpgan import SecCMP_comm

@mpc.run_multiprocess(world_size=2)
def run_SecReLU():
    # 创建加密张量
    u_enc = crypten.cryptensor([[-4,1,9]], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor(0, ptype=crypten.mpc.arithmetic)
    # S1 == R0, S2 == R1
    # 1) 2)
    # 提取当前参与方的 _tensor 属性 ，即提取份额
    u_shared = u_enc._tensor.data
    v_shared = v_enc._tensor.data
    f = SecCMP_comm.SecCmp(u_enc, v_enc)

    u = torch.tensor([[-4,1,9]])
    v = torch.tensor([[3,2,1]])

    result_torch = torch.relu(u)
    print("result_torch:", result_torch)
    f_enc = crypten.cryptensor(f, ptype=crypten.mpc.arithmetic)
    SecCMP = u_enc * f_enc

    # SecCMP = u_enc * f
    x = torch.tensor([[0, 1, 1]])
    z = u_enc * x

    crypten.print("f:", f)
    crypten.print("f'type:", f.dtype)
    crypten.print("SecCMP:", SecCMP.get_plain_text())
    crypten.print("x'type:", x.dtype)
    crypten.print(f"z_dec:{z.get_plain_text()}")

run_SecReLU()



