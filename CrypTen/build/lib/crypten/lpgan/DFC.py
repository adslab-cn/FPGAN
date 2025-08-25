import torch

def DFC_element_signed(u, l):
    """
    Data Format Conversion (DFC) algorithm to convert an integer u into an l-bit signed binary format for a single element.

    Parameters:
    u (int or torch.Tensor): The integer to convert.
    l (int): The length of the signed binary format.

    Returns:
    list: A list of bits representing the l-bit signed binary format of u.
    """
    if isinstance(u, torch.Tensor):
        u = u.item()

    bits = [0] * l
    abs_u = abs(u)

    # Convert the absolute value to binary
    for i in range(l - 1):
        bits[i] = (abs_u >> i) & 1

    # Set the sign bit
    sign_bit = 1 if u < 0 else 0
    bits[-1] = sign_bit

    return bits

def DFC_element_unsigned(u, l):
    """
    Data Format Conversion (DFC) algorithm to convert an integer u into an (l-1)-bit unsigned binary format for a single element.

    Parameters:
    u (int or torch.Tensor): The integer to convert.
    l (int): The length of the signed binary format, (l-1) will be used for unsigned format.

    Returns:
    list: A list of bits representing the (l-1)-bit unsigned binary format of u.
    """
    if isinstance(u, torch.Tensor):
        u = u.item()

    bits = [0] * (l - 1)
    abs_u = abs(u)

    for i in range(l - 1):
        bits[i] = (abs_u >> i) & 1

    return bits

def DFC(tensor, l):
    """
    Apply DFC algorithm to each element of a multi-dimensional tensor and return both signed and unsigned binary representations.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    l (int): The length of the signed binary format for each element.

    Returns:
    tuple: Two tensors, one with signed binary representation and one with unsigned binary representation.
    """
    signed_result = []
    unsigned_result = []

    for elem in tensor.flatten():
        signed_binary_repr = DFC_element_signed(elem, l)
        signed_result.append(signed_binary_repr)
        
        unsigned_binary_repr = DFC_element_unsigned(elem, l)
        unsigned_result.append(unsigned_binary_repr)
    
    signed_result_tensor = torch.tensor(signed_result).view(*tensor.shape, l)
    unsigned_result_tensor = torch.tensor(unsigned_result).view(*tensor.shape, l - 1)

    return signed_result_tensor, unsigned_result_tensor

def extract_sign_bits(signed_binary_tensor):
    """
    Extract the sign bits from a signed binary representation tensor.

    Parameters:
    signed_binary_tensor (torch.Tensor): The tensor with signed binary representation.

    Returns:
    torch.Tensor: A tensor containing only the sign bits.
    """
    sign_bits = signed_binary_tensor[..., -1]
    return sign_bits