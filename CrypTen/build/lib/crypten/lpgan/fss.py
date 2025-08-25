import funshade
import numpy as np
import torch
import crypten
import crypten.communicator as comm
import torch.distributed as dist
import crypten.mpc as mpc

crypten.init()

K = 1       # Number of elements in the input vector
theta = 0    # Threshold to compare it with

@mpc.run_multiprocess(world_size=2)
def run(a):
    # Create integer vector of length K
    rng = np.random.default_rng(seed=42)
    a = a  # Tensor a
    # z = crypten.cryptensor(torch.tensor([a], dtype=torch.int64))  # Input vector z
    z = a._tensor.data

    ## Offline Phase
    r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, theta)

    ## Online Phase
    rank = comm.get().get_rank()
    if rank == 0:
        z_0 = z + r_in0
        z_0 = z_0.numpy().astype(np.int32)
        z_0 = torch.tensor(z_0)  # Convert to torch.Tensor
        z_1 = torch.zeros_like(z_0)  # Initialize z_1 to zeros
    else:
        z_1 = z + r_in1
        z_1 = z_1.numpy().astype(np.int32)
        z_1 = torch.tensor(z_1)  # Convert to torch.Tensor
        z_0 = torch.zeros_like(z_1)  # Initialize z_0 to zeros
    dist.barrier()
    dist.broadcast(z_0, src=0)
    dist.broadcast(z_1, src=1)

    # Compute the comparison to the threshold theta
    if rank == 0:
        o_j_1 = funshade.eval_sign(K, rank, k0, z_0.numpy(), z_1.numpy())
        o_j_1 = torch.tensor(o_j_1)  # Convert to torch.Tensor
        o_j_2 = torch.zeros_like(o_j_1)
    else:
        o_j_2 = funshade.eval_sign(K, rank, k1, z_1.numpy(), z_0.numpy())
        o_j_2 = torch.tensor(o_j_2)  # Convert to torch.Tensor
        o_j_1 = torch.zeros_like(o_j_2)

    dist.barrier()
    dist.broadcast(o_j_1, src=0)
    dist.broadcast(o_j_2, src=1)

    o = o_j_1 + o_j_2
    crypten.print("o:", o)
    # Check the final result

    o_ground = (a >= theta)  # Convert o_ground to integer
    crypten.print("o_ground:", o_ground)
    crypten.print("FSS gate executed correctly")

run()
