#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch

from .module import Module


class _Loss(Module):
    """
    Base criterion class that mimics Pytorch's Loss.
    """

    def __init__(self, reduction="mean", skip_forward=False):
        super(_Loss, self).__init__()
        if reduction != "mean":
            raise NotImplementedError("reduction %s not supported")
        self.reduction = reduction
        self.skip_forward = skip_forward

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattribute__(self, name):
        if name != "forward":
            return object.__getattribute__(self, name)

        def forward_function(*args, **kwargs):
            """Silently encrypt Torch tensors if needed."""
            if self.encrypted or any(
                isinstance(arg, crypten.CrypTensor) for arg in args
            ):
                args = list(args)
                for idx, arg in enumerate(args):
                    if torch.is_tensor(arg):
                        args[idx] = crypten.cryptensor(arg)
            return object.__getattribute__(self, name)(*tuple(args), **kwargs)

        return forward_function


class MSELoss(_Loss):
    r"""
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = (x_n - y_n)^2,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return (x - y).square().mean()


class L1Loss(_Loss):
    r"""
    Creates a criterion that measures the mean absolute error between each element in
    the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = \left | x_n - y_n \right |,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return (x - y).abs().mean()


# class BCELoss(_Loss):
#     r"""
#     Creates a criterion that measures the Binary Cross Entropy
#     between the prediction :math:`x` and the target :math:`y`.

#     The loss can be described as:

#     .. math::
#         \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
#         l_n = - \left [ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right ],

#     where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
#     arbitrary shapes with a total of :math:`n` elements each.

#     This is used for measuring the error of a reconstruction in for example
#     an auto-encoder. Note that the targets :math:`y` should be numbers
#     between 0 and 1.
#     """  # noqa: W605

#     def forward(self, x, y):
#         assert x.size() == y.size(), "input and target must have the same size"
#         return x.binary_cross_entropy(y, skip_forward=self.skip_forward)

# class BCELoss(_Loss):
#     def __init__(self, eps=2 ** -16, skip_forward=False):  # eps = 2^-8
#         super(BCELoss, self).__init__()
#         self.eps = eps
#         self.skip_forward = skip_forward

#     def forward(self, x, y):
#         assert x.size() == y.size(), "input and target must have the same size"
        
#         # 判断输入值的范围
#         is_negative = (x.min()).ibdcf()
#         if is_negative:  # 解密后获取布尔值
#             min_val = self.eps
#             max_val = 1 - self.eps
#         else:
#             min_val = -1 + self.eps
#             max_val = - self.eps


#         # # 在密文中实现 clamp 功能
#         # x = self._clamp(x, min=min_val, max=max_val)
        
#         # 计算交叉熵损失
#         if self.skip_forward:
#             return x.new(0)
#         return x.binary_cross_entropy(y, skip_forward=self.skip_forward)

#     def _clamp(self, x, min, max):
#         """
#         在密文中实现 clamp 功能，将 x 限制在 [min, max] 范围内。
#         """
#         # 将 x 小于 min 的值替换为 min
#         x = crypten.where(x < min, crypten.cryptensor(min), x)
#         # 将 x 大于 max 的值替换为 max
#         x = crypten.where(x > max, crypten.cryptensor(max), x)
#         return x
class BCELoss(_Loss):
    def __init__(self, eps=2 ** -16, skip_forward=False):
        super(BCELoss, self).__init__()
        self.eps = eps
        self.skip_forward = skip_forward
        
    # @track_network_traffic
    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        
        # 在 DCGAN 中，x 经过 Sigmoid，范围应为 [0, 1]
        # 限制 x 在 [eps, 1 - eps] 以避免 log(0) 或 log(1) 问题
        min_val = self.eps
        max_val = 1 - self.eps

        # 在密文中实现 clamp 功能
        x = self._clamp(x, min=min_val, max=max_val)
        
        # 计算交叉熵损失
        if self.skip_forward:
            return x.new(0)
        return x.binary_cross_entropy(y, skip_forward=self.skip_forward)

    def _clamp(self, x, min, max):
        """
        在密文中实现 clamp 功能，将 x 限制在 [min, max] 范围内。
        """
        # 将 x 小于 min 的值替换为 min
        x = crypten.where(x < min, crypten.cryptensor(min), x)
        # 将 x 大于 max 的值替换为 max
        x = crypten.where(x > max, crypten.cryptensor(max), x)
        return x
    
device = torch.device('cpu')
from crypten.lpgan.Secfunctions import timeit, track_network_traffic,get_network_bytes, calculate_network_traffic
class ESBCELoss(_Loss):
    def __init__(self, eps=2**-16, skip_forward=False, p0=0.1, q0=0.9, delta=0.0001):
        super(ESBCELoss, self).__init__()
        self.eps = eps
        self.skip_forward = skip_forward
        self.delta = delta
        
        # 初始化裁剪边界，确保在 [eps, 1-eps] 内
        self.p = max(p0, eps)
        self.q = min(q0, 1 - eps)
        assert self.p < self.q, "初始 p 必须小于 q"
        # assert (self.q).ibdcf(self.p) , "初始 p 必须小于 q"
        
    def _clamp_input(self, x):
        """将输入限制在 [eps, 1-eps] 内"""
        min_val = crypten.cryptensor(self.eps)
        max_val = crypten.cryptensor(1 - self.eps)
        # ==============LPGAN and CrypTen==============
        # x_clamped = crypten.where(x < min_val, min_val, x)
        # x_clamped = crypten.where(x > max_val, max_val, x)
        # ==============LPGAN==============

        # print("min_val.ibdcf(alpha=x.get_plain_text())'size:", (min_val.ibdcf(alpha=x.get_plain_text())).size())
        # x_clamped = crypten.where(min_val.ibdcf(alpha=x.get_plain_text()), min_val, x.reshape((min_val.ibdcf(alpha=x.get_plain_text())).shape))
        # print("x.ibdcf(alpha=max_val.get_plain_text())'size:", (x.ibdcf(alpha=max_val.get_plain_text())).size())
        # print("max_val'size:", max_val.size())

        # ==============FSSGAN==============
        x_clamped = crypten.where(-(x.ibdcf(alpha=min_val.get_plain_text())).reshape(x.shape), min_val, x)
        x_clamped = crypten.where((x.ibdcf(alpha=max_val.get_plain_text())).reshape(x.shape), max_val, x)
        # ==============FSSGAN==============

        # x_clamped = crypten.where((x.dcf(alpha=min_val.get_plain_text())).reshape(x.shape), min_val, x)
        # x_clamped = crypten.where(-(x.dcf(alpha=max_val.get_plain_text())).reshape(x.shape), max_val, x)

        return x_clamped
    
    def _dynamic_clip(self, x):
        """使用 ibdicf 实现动态裁剪"""
        p_tensor = crypten.cryptensor(self.p, device=device)
        q_tensor = crypten.cryptensor(self.q, device=device)
        eps_tensor = crypten.cryptensor(self.eps, device=device)
        one_tensor = crypten.cryptensor(1.0, device=device)
        
        # 使用 ibdicf 构造掩码
        # # ge_p = x.ibdicf()  # x >= p
        # # le_q = x.ibdicf()  # x <= q
        # ge_p = x.dicf()  # x >= p
        # le_q = x.dicf()  # x <= q

        ge_p = x.ibdicf(self.p, 1)  # x >= p
        le_q = x.ibdicf(self.eps, self.q)  # x <= q
        # ge_p = x.dicf(self.p, 1)  # x >= p
        # le_q = x.dicf(self.eps, self.q)  # x <= q
        
        mask_low = crypten.cryptensor(1 - ge_p, device=device)    # x < p
        mask_high = crypten.cryptensor(1 - le_q, device=device)   # x > q
        mask_in = crypten.cryptensor(ge_p, device=device) * crypten.cryptensor(le_q, device=device)  # p <= x <= q

        
        # 应用裁剪逻辑
        # print("mask_low'size:", mask_low.size())
        # print("mask_high'size:", mask_high.size())
        # print("mask_in'size:", mask_in.size())
        # print("x'size:", x.size())
        # print("p_tensor'size:", p_tensor.size())
        
        x_clipped = mask_low * p_tensor + mask_high * q_tensor + mask_in * (x.to('cpu')).view(-1)
        # x_clipped = mask_low * p_tensor + mask_high * q_tensor + mask_in * (x.to('cpu'))
        return x_clipped
    
    def adjust_bounds(self, loss_increased):
        """根据损失变化调整裁剪边界"""
        if loss_increased:
            self.p = min(self.p + self.delta, 1 - self.eps)
            self.q = max(self.q - self.delta, self.eps)
        else:
            self.p = max(self.p - self.delta, self.eps)
            self.q = min(self.q + self.delta, 1 - self.eps)
        self.p, self.q = sorted([self.p, self.q])
        self.p = max(self.p, self.eps)
        self.q = min(self.q, 1 - self.eps)
    
    @timeit
    @track_network_traffic
    def forward(self, x, y):
        assert x.size() == y.size(), "输入和目标尺寸不匹配"
        
        if self.skip_forward:
            return crypten.cryptensor(0.0)
            
        # 1. 输入限制
        x_clamped = self._clamp_input(x)
        
        # 2. 动态裁剪
        x_clipped = self._dynamic_clip(x_clamped)
        
        # # 3. 计算损失
        # log_pos = x_clipped.log()
        # log_neg = (1 - x_clipped).log()
        # y = y.view(-1)
        # loss_terms = y * log_pos + (1 - y) * log_neg
        # loss = -loss_terms.mean()
        loss = x_clipped.binary_cross_entropy(y.reshape(x_clipped.shape), skip_forward=self.skip_forward)
        
        return loss
    
class AdaptiveBCELoss(Module):
    def __init__(self, precision_bits=16, safety_margin=4, skip_forward=False):
        super(AdaptiveBCELoss, self).__init__()
        self.precision_bits = precision_bits      # 小数位数 (L)
        self.safety_margin = safety_margin        # 安全余量位
        self.skip_forward = skip_forward
        
        # 动态计算裁剪边界
        self.eps = 2 ** -(self.precision_bits + self.safety_margin)
        
        # 预加密的裁剪边界常量（优化性能）
        self.eps_enc = crypten.cryptensor(self.eps)
        self.one_eps_enc = crypten.cryptensor(1 - self.eps)
        self.neg_one_eps_enc = crypten.cryptensor(-1 + self.eps)
        self.neg_eps_enc = crypten.cryptensor(-self.eps)

    def forward(self, x, y):
        assert x.size() == y.size(), "Input and target must have the same size"
        
        # -------- 安全范围检查 (Secure Range Check) --------
        # 使用安全协议判断是否全为非负数
        is_negative = ((x.min()).ge(0)).get_plain_text()
        
        # -------- 动态裁剪 (Secure Clamping) --------
        if is_negative:
            min_val = self.neg_one_eps_enc
            max_val = self.neg_eps_enc
        else:
            min_val = self.eps_enc
            max_val = self.one_eps_enc
        
        x_clamped = self._secure_clamp(x, min_val, max_val)
        
        # -------- 损失计算 (Secure Loss Computation) --------
        if self.skip_forward:
            return x.new(0)
        
        # 安全计算 log(x + eps) 和 log(1 - x + eps)
        x_plus_eps = x_clamped + self.eps_enc
        one_minus_x_plus_eps = 1 - x_clamped + self.eps_enc
        
        log_x = x_plus_eps.log(input_in_01=True)
        log_1_x = one_minus_x_plus_eps.log(input_in_01=True)
        
        loss_elements = - (y * log_x + (1 - y) * log_1_x)
        return loss_elements.mean()

    def _secure_clamp(self, x, min_val, max_val):
        """加密环境下的安全裁剪操作"""
        # 使用广播确保形状一致
        # min_val = min_val.expand_as(x)
        # max_val = max_val.expand_as(x)
        
        # 逐元素比较和替换
        x_clamped = crypten.where(x < min_val, min_val, x)
        x_clamped = crypten.where(x_clamped > max_val, max_val, x_clamped)
        return x_clamped
    
class CrossEntropyLoss(_Loss):
    r"""
    Creates a criterion that measures cross-entropy loss between the
    prediction :math:`x` and the target :math:`y`. It is useful when
    training a classification problem with `C` classes.

    The prediction `x` is expected to contain raw, unnormalized scores for each class.

    The prediction `x` has to be a Tensor of size either :math:`(N, C)` or
    :math:`(N, C, d_1, d_2, ..., d_K)`, where :math:`N` is the size of the minibatch,
    and with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    target `y` for each value of a 1D tensor of size `N`.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log \left(
        \frac{\exp(x[class])}{\sum_j \exp(x[j])} \right )
        = -x[class] + \log \left (\sum_j \exp(x[j]) \right)

    The losses are averaged across observations for each batch

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape.
    """  # noqa: W605

    def forward(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        assert x.size() == y.size(), "input and target must have the same size"
        return x.cross_entropy(y, skip_forward=self.skip_forward)


class BCEWithLogitsLoss(_Loss):
    r"""
    This loss combines a Sigmoid layer and the BCELoss in one single class.

    The loss can be described as:

    .. math::
        p = \sigma(x)

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log p_n + (1 - y_n) \cdot \log (1 - p_n) \right ],

    This is used for measuring the error of a reconstruction in for example an
    auto-encoder. Note that the targets t[i] should be numbers between 0 and 1.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.binary_cross_entropy_with_logits(y, skip_forward=self.skip_forward)


class RAPPORLoss(_Loss):
    r"""
    This loss computes the BCEWithLogitsLoss with corrections applied to account
    for randomized response, where the input `alpha` represents the probability
    of flipping a label.

    The loss can be described as:

    .. math::
        p = \sigma(x)

    .. math::
        r = \alpha * p + (1 - \alpha) * (1 - p)

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log r_n + (1 - y_n) \cdot \log (1 - r_n) \right ],

    This is used for measuring the error of a reconstruction in for example an
    auto-encoder. Note that the targets t[i] should be numbers between 0 and 1.
    """

    def __init__(self, alpha, reduction="mean", skip_forward=False):
        super(RAPPORLoss, self).__init__(reduction=reduction, skip_forward=skip_forward)
        self.alpha = alpha

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.rappor_loss(y, self.alpha, skip_forward=self.skip_forward)
