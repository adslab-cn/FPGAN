import torch
from crypten.lpgan import DFC

# Example usage
x_div = torch.tensor([[-8, 7], [3, -3]])
l = 8

signed_x_div_binary, unsigned_x_div_binary = DFC.DFC(x_div, l)

print("Input tensor:")
print(x_div)
print(f"\nSigned binary representation ({l} bits for each element):")
print(signed_x_div_binary)
print(f"\nUnsigned binary representation ({l-1} bits for each element):")
print(unsigned_x_div_binary)

# Example usage
x_div = torch.tensor([[7], [-2]])
l = 8  # Using 8 bits for signed representation

# Convert the tensor to l-bit signed binary format
signed_x_div_binary, unsigned_x_div_binary = DFC.DFC(x_div, l)
# Extract sign bits from the signed binary representation
sign_bits = DFC.extract_sign_bits(signed_x_div_binary)

print("\nSign bits:")
print(sign_bits)