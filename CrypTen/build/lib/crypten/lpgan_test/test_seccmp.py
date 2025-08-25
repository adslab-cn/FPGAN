import torch
import crypten
import crypten.mpc as mpc
from crypten.lpgan import SecCMP

@mpc.run_multiprocess(world_size=2)
def run_SecCmp():
    # 创建加密张量
    u_enc = crypten.cryptensor([[1,1,91]], ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor([[1,1,90]], ptype=crypten.mpc.arithmetic)
    # S1 == R0, S2 == R1
    # 1) 2)
    # 提取当前参与方的 _tensor 属性 ，即提取份额
    u_shared = u_enc._tensor.data
    v_shared = v_enc._tensor.data
    f = SecCMP.SecCmp(u_shared, v_shared)
    crypten.print("f_cmp:", f)
run_SecCmp()