#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import crypten
import torch
from crypten.config import cfg

import math
import random
import crypten
import torch
import torch.distributed as dist
from crypten import mpc
import crypten.communicator as comm 
from crypten.lpgan.SecSRC import SRC
from crypten.lpgan.Secfunctions import SecAdd
import time

__all__ = [
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    # "SISqrt",
    # "SSqrt",
    "inv_sqrt_goldschmidt",
    "sqrt_goldschmidt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "softmax",
    "log_softmax",
]

import traceback
# Iterative methods:
def exp(self):
    r"""Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.
    """  # noqa: W605
    iters = cfg.functions.exp_iterations
    # print("===========exp=============\n")
    result = 1 + self.div(2**iters)
    for _ in range(iters):
        result = result.square()

    return result


def log(self, input_in_01=False):
    r"""
    Approximates the natural logarithm using 8th order modified
    Householder iterations. This approximation is accurate within 2% relative
    error on [0.0001, 250].

    Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

    .. math::

        y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the domain [0, 1],
            causing the function optimize for this domain. This is useful for computing
            log-probabilities for entropy functions.

            We shift the domain of convergence by a constant :math:`a` using the following identity:

            .. math::

                \ln{u} = \ln {au} - \ln{a}

            Since the domain of convergence for CrypTen's log() function is approximately [1e-4, 1e2],
            we can set :math:`a=100`.

    Configuration parameters:
        iterations (int): number of Householder iterations for the approximation
        exp_iterations (int): number of iterations for limit approximation of exp
        order (int): number of polynomial terms used (order of Householder approx)
    """
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    iterations = cfg.functions.log_iterations
    exp_iterations = cfg.functions.log_exp_iterations
    order = cfg.functions.log_order

    term1 = self.div(120)
    term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
    y = term1 - term2 + 3.0

    # 8th order Householder iterations
    with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
        for _ in range(iterations):
            h = 1 - self * exp(-y)
            y -= h.polynomial([1 / (i + 1) for i in range(order)])
    return y


# def log(self, input_in_01=False):
#     """
#     使用 Maclaurin 级数近似自然对数，符合 SecLog 协议。
    
#     Args:
#         input_in_01 (bool): 如果输入在 [0, 1] 范围内，进行适当的缩放处理。
    
#     Returns:
#         float: 计算得到的自然对数。
#     """
#     # 对输入进行域变换，如果输入较大，将其缩放到合理范围
#     if input_in_01:
#         input_in = self.mul(100)  # 例如，缩放因子可以是 100
#         return log(input_in) - 4.605170  # ln(100) ≈ 4.605170

#     # 使用 log(u) = log(u / a) + log(a) 的恒等式来扩展域
#     a = 100.0
#     one_tensor = torch.tensor(1.0)
#     log_a = torch.tensor(4.605170)

#     # 判断 self 是否大于 1
#     is_greater_than_one = self.gt(one_tensor)  # 比较操作
#     # 计算 scaled_input = self / a
#     scaled_input = self.div(a)

#     # 使用加密选择：如果大于 1，使用 scaled_input，否则使用 self
#     input_in = is_greater_than_one * scaled_input + (1 - is_greater_than_one) * self
#     # 计算 log 的偏移量
#     result_offset = is_greater_than_one * log_a

#     # 初始化估计值
#     x = input_in.sub(1.0)  # x = u - 1

#     # 进行 Maclaurin 级数近似
#     order = 8  # 使用的级数项数
#     result = x  # 第一项 x
#     sign = -1   # 符号交替

#     # 当前项的初始值为 x
#     current_term = x

#     # 逐项计算级数和
#     for i in range(2, order + 1):
#         # 计算 x ** i，这里使用 mul 替代平方操作
#         current_term = current_term.mul(x)  # 计算 x ** i
        
#         # 除以 i，这里使用 div 替代除法
#         term = current_term.div(i)
        
#         # 更新结果，交替加减
#         result = result.add(term.mul(sign))
        
#         # 交替符号
#         sign = -sign

#     # 返回结果加上偏移量
#     return result.add(result_offset)  # 将缩放的补偿加回结果


def reciprocal(self, input_in_01=False):
    r"""
    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

    Configuration params:
        reciprocal_method (str):  One of 'NR' or 'log'.
        reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method
        reciprocal_log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method
        reciprocal_all_pos (bool): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input.
        reciprocal_initial (tensor): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """

    rank = comm.get().get_rank()
    # crypten.print(f"========in reciprocal=====\nRank{rank}self:{self}", in_order=True)
    pos_override = {"functions.reciprocal_all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64)).mul(64)
        return rec

    # Get config options
    method = cfg.functions.reciprocal_method
    all_pos = cfg.functions.reciprocal_all_pos
    initial = cfg.functions.reciprocal_initial

    if not all_pos:
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos)

    if method == "NR":
        nr_iters = cfg.functions.reciprocal_nr_iters
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result += result - result.square().mul_(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal_log_iters
        with cfg.temp_override({"functions.log_iters": log_iters}):
            return exp(-log(self))
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters

    # Initialize using decent approximation
    if initial is None:
        y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
        y -= self.div(1024)
    else:
        y = initial

    # Newton Raphson iterations for inverse square root
    for _ in range(iters):
        y = y.mul_(3 - self * y.square()).div_(2)
    return y

def inv_sqrt_goldschmidt(self, iters=8):
    """
    Computes the inverse square root using the Goldschmidt method without relying on rsqrt().
    
    Args:
        iters (int): Number of iterations for Goldschmidt refinement.
    
    Returns:
        torch.Tensor: Approximation of the inverse square root.
    """
    # 1. 初始值估计
    # 使用指数近似： exp(-x / 2) * c 是一种有效的粗略估计
    y = exp(self.div(2).neg()).mul(0.5).add(0.5)  # y ≈ 1 / sqrt(x) 初始值

    # 2. Goldschmidt迭代
    z = self.mul(y.square())  # z = x * y^2

    for _ in range(iters):
        y = y.mul(1.5 - z.mul(0.5))  # y = y * (1.5 - 0.5 * z)
        z = self.mul(y.square())  # 更新 z

    return y

def sqrt_goldschmidt(self, iters=8):
    """
    Computes the square root of the input using the Goldschmidt method.

    Args:
        iters (int): Number of iterations to run.
    
    Returns:
        torch.Tensor: Approximation of the square root.
    """
    # Step 1: Initial approximation for sqrt(x)
    y = self.div(2).add(0.5)  # Approximation: y ≈ x / 2 + 0.5

    # Step 2: Goldschmidt iteration
    for _ in range(iters):
        z = self.div(y.square())  # z = x / y^2
        y = y.mul(1.5 - z.mul(0.5))  # y = y * (1.5 - 0.5 * z)

    return y


# def inv_sqrt(self):
#     """
#     Input:u1, u2; Iterative number: I
#     Output:f1, f2
#     """
#     I = 10
#     crypten.print("===哈哈哈SISqrt===\n")
#     from crypten.mpc.primitives import ArithmeticSharedTensor
#     self = self._tensor.data
#     m, p = SRC(self)
#     rank = comm.get().get_rank()
#     if rank == 0:
#         alpha = -0.8099868542
#         beta = 1.787727479
#         y = alpha * m + beta
#         h = 1/2 * (alpha * m + beta)
        
#     else:
#         alpha = -0.8099868542
#         beta = 1.787727479
#         y = alpha * m
#         h = 1/2 * (alpha * m)

#     dist.barrier()
#     m = SecAdd(m)
#     y = SecAdd(y)
#     h = SecAdd(h)

#     m = ArithmeticSharedTensor(m)
#     y = ArithmeticSharedTensor(y)
    
#     # 4) SecMul(m, y0)
#     g = m * y

#     # 5) iterative
#     i = 0
#     h = ArithmeticSharedTensor(h)

#     while i < I:
#         # 6) SecMul(g,h)
#         r = g * h
#         r = r._tensor.data.double()
#         # 7)
#         if rank == 0:
#             r = 3/2 - r/2**16
#         else:
#             r = -r/2**16
#         dist.barrier()
#         r = SecAdd(r)
#         r = r
        
#         r = ArithmeticSharedTensor(r)
#         # 8) g=SecMul(g, r), h=SecMul(h, r)
#         g = g * r
#         h = h * r
#         i += 1

#     h = h._tensor.data
#     # 9) a = 1 if p is even else a = 2**(1/2)
#     a = torch.where(p % 2 == 0, torch.tensor(1.0), torch.tensor(math.sqrt(2)))
#     # 10) p_ = p/2(向下取整)
#     p_ = torch.floor(p/2)
#     # 11) 12) 13)
#     if rank == 0:
#         f = 2/(a * 2**(p_)) * h
#     else:
#         f = 2/(a * 2**(p_)) * h
#     f = crypten.cryptensor(SecAdd(f)/2**16)
#     return f
    
# def sqrt(self, I, l=(2**32)):
#     """
#     Input:u1, u2; Iterative number: I
#     Output:f1, f2
#     """
#     I = I
#     # crypten.print("===SSqrt===\n")
#     from crypten.mpc.primitives import ArithmeticSharedTensor
#     from crypten.lpgan.Secfunctions import get_network_bytes, calculate_network_traffic
#     net_io_before = get_network_bytes()
#     start_time = time.time()

#     self = self._tensor.data
#     m, p = SRC(self)
#     rank = comm.get().get_rank()
#     if rank == 0:
#         alpha = -0.8099868542
#         beta = 1.787727479
#         y = alpha * m + beta
#         h = 1/2 * (alpha * m + beta)
#     else:
#         alpha = -0.8099868542
#         beta = 1.787727479
#         y = alpha * m
#         h = 1/2 * (alpha * m)
#     dist.barrier()
#     m = SecAdd(m)
#     y = SecAdd(y)
#     h = SecAdd(h)

#     m = ArithmeticSharedTensor(m)
#     y = ArithmeticSharedTensor(y)

#     # 4) SecMul(m, y0)
#     g = m * y
#     net_io_after = get_network_bytes()
#     init_time = time.time() - start_time
#     traffic_diff_init = calculate_network_traffic(net_io_before, net_io_after)
    
# ########################################
#     # 5) iterative
#     net_io_before = get_network_bytes()
#     start_time = time.time()

#     i = 0
#     h = ArithmeticSharedTensor(h)

#     while i < I:
#         # 6) SecMul(g,h)
#         # r = g * h
#         r = g.mul(h)

#         r = r._tensor.data.double()
#         # 7)
#         if rank == 0:
#             r = 3/2 - r/2**16
#         else:
#             r = -r/2**16

#         dist.barrier()
#         r = SecAdd(r)
#         r = ArithmeticSharedTensor(r)
#         # 8) g=SecMul(g, r), h=SecMul(h, r)
#         g = g * r
#         h = h * r
#         # g = g.mul(r,l)
#         # h = h.mul(r,l)
#         i += 1

#     g = g._tensor.data
#     # 9) a = 1 if p is even else a = 2**(1/2)
#     a = torch.where(p % 2 == 0, torch.tensor(1.0), torch.tensor(math.sqrt(2)))
#     # 10) p_ = p/2(向下取整)
#     p_ = torch.floor(p/2)
#     # 11) 12) 13)
#     if rank == 0:
#         f = a * 2**(p_) * g
#     else:
#         f = a * 2**(p_) * g
#     # f = crypten.cryptensor(SecAdd(f)/2**16)
#     # f = SecAdd(f)/2**16
#     f = comm.get().all_reduce(f)/2**16
#     net_io_after = get_network_bytes()
#     itera_time = time.time() - start_time
#     traffic_diff_itera = calculate_network_traffic(net_io_before, net_io_after)
#     return f, traffic_diff_init, traffic_diff_itera, init_time, itera_time
    
def sqrt(self):
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run
        sqrt_initial (tensor): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return inv_sqrt(self).mul_(self)


def _eix(self):
    r"""Computes e^(i * self) where i is the imaginary unit.
    Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
    """
    iterations = cfg.functions.trig_iterations

    re = 1
    im = self.div(2**iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im.square()
    im *= 2

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul_(re)
        im._tensor *= 2
        re = a2 - b2

    return re, im


def cossin(self):
    r"""Computes cosine and sine of input via exp(i * x).

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return self._eix()


def cos(self):
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[0]


def sin(self):
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[1]


# Logistic Functions
def sigmoid(self):
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    If a valid method is given, this function will compute sigmoid
        using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with
        truncation and uses the identity:

    .. math::
        \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

    "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
        the reciprocal

    """  # noqa: W605
    method = cfg.functions.sigmoid_tanh_method

    if method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sign = 1 - 2 * ltz

        pos_input = self.mul(sign)
        denominator = pos_input.neg().exp().add(1)

        # TODO: Set these with configurable parameters
        with cfg.temp_override(
            {
                "functions.exp_iterations": 9,
                "functions.reciprocal_nr_iters": 3,
                "functions.reciprocal_all_pos": True,
                "functions.reciprocal_initial": 0.75,
            }
        ):
            pos_output = denominator.reciprocal()

        result = pos_output.where(1 - ltz, 1 - pos_output)
        # TODO: Support addition with different encoder scales
        # result = pos_output + ltz - 2 * pos_output * ltz
        return result
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")


def tanh(self):
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    If a valid method is given, this function will compute tanh using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with truncation.

    .. math::
        tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

    where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
    The approximation is truncated to +/-1 outside [-1, 1].

    Args:
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    """
    method = cfg.functions.sigmoid_tanh_method

    if method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh_terms
        coeffs = crypten.common.util.chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return out.hardtanh()
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def _chebyshev_polynomials(self, terms):
    r"""Evaluates odd degree Chebyshev polynomials at x

    Chebyshev Polynomials of the first kind are defined as

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        self (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return crypten.stack(polynomials)


def erf(tensor):
    r"""
    Approximates the error function of the input tensor using a Taylor approximation.
    """
    iters = cfg.functions.erf_iterations

    output = tensor.clone()
    for n in range(1, iters + 1):
        multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
        output = output.add(tensor.pos_pow(2 * n + 1).mul(multiplier))
    return output.mul(2.0 / math.sqrt(math.pi))
    # NOTE: This approximation is not unstable for large tensor values.


def softmax(self, dim, **kwargs):
    r"""Compute the softmax of a tensor's elements along a given dimension"""
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.ones_like((self.data)))

    if self.size(dim) == 1:
        return self.new(torch.ones_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    numerator = logits.exp()
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
    return numerator * inv_denominator


def log_softmax(self, dim, **kwargs):
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.zeros((), device=self.device))

    if self.size(dim) == 1:
        return self.new(torch.zeros_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - normalize_term.log()
    return result
