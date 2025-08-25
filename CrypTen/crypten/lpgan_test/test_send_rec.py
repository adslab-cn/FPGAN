import crypten
import torch
import torch.distributed as dist
from crypten import mpc
import crypten.communicator as comm 

crypten.init()

@mpc.run_multiprocess(world_size=2)
def main():
    # Initialize the distributed environment
    rank = crypten.communicator.get().get_rank()

    # Create plaintext tensors
    u = torch.tensor([[10]], dtype=torch.float32)
    v = torch.tensor([[1]], dtype=torch.float32)

    # Encrypt tensors
    u_enc = crypten.cryptensor(u, ptype=crypten.mpc.arithmetic)
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)

    # Get shares for each rank
    u_shared = u_enc._tensor.data
    v_shared = v_enc._tensor.data
    crypten.print(f"rank {rank} - u_shared: {u_shared}", in_order=True)
    crypten.print(f"rank {rank} - v_shared: {v_shared}", in_order=True)

    # Perform independent computation on each rank
    if rank == 0:
        a = u_shared - v_shared
        crypten.print(f"Node {rank} result: {a}", in_order=True)
    elif rank == 1:
        b = v_shared - u_shared
        crypten.print(f"Node {rank} result: {b}", in_order=True)

    # Synchronize all processes
    dist.barrier()

    # Broadcast to ensure both nodes receive the results
    if rank == 0:
        a_tensor = a
        b_tensor = torch.zeros_like(a_tensor)
    else:
        a_tensor = torch.zeros_like(u_shared)
        b_tensor = b

    dist.broadcast(a_tensor, src=0)
    dist.broadcast(b_tensor, src=1)

    # Print results
    crypten.print(f"Rank {rank} - a_tensor: {a_tensor}", in_order=True)
    crypten.print(f"Rank {rank} - b_tensor: {b_tensor}", in_order=True)

    # Convert tensors to cryptensors
    a = crypten.cryptensor(a_tensor)
    b = crypten.cryptensor(b_tensor)
    crypten.print(f"Rank {rank} a: {a}", in_order=True)
    crypten.print(f"Rank {rank} b: {b}", in_order=True)

    # Synchronize before sending/receiving tensor
    dist.barrier()

    # Node 0 sends a tensor to Node 1
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # Example tensor
        print(f"Node {rank} prepared tensor to send: {tensor}")
        comm.get().send(tensor, dst=1)
        print(f"Node {rank} sent tensor: {tensor}")

    # Node 1 receives the tensor from Node 0
    elif rank == 1:
        tensor = torch.empty(3, dtype=torch.float32)  # Create an empty tensor with the same shape and dtype
        print(f"Node {rank} prepared to receive tensor.")
        comm.get().recv(tensor, src=0)
        print(f"Node {rank} received tensor: {tensor}")

    # Final synchronization to ensure all processes complete
    dist.barrier()

# Run the main function
main()
