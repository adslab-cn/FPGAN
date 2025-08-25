#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch

from .optimizer import Optimizer

class Adam(Optimizer):
    r"""Implements Adam optimization algorithm.
    
    Adam is an adaptive learning rate optimization algorithm based on
    adaptive estimates of lower-order moments. It combines the advantages
    of both Adagrad and RMSProp.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_threshold (float, optional): imposes a threshold on the magnitude of gradient values.
            Gradient values with magnitude above the threshold will be replaced with 0.
    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, 
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 grad_threshold=None):
        if not isinstance(lr, (int, float)) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(betas, tuple) or len(betas) != 2 or not all(isinstance(b, (int, float)) for b in betas):
            raise ValueError("Invalid betas: {}".format(betas))
        if not isinstance(eps, (int, float)) or eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        
        # Compute thresholding based on square value since abs is more expensive
        self.square_threshold = grad_threshold
        if self.square_threshold is not None:
            self.square_threshold *= self.square_threshold

        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with crypten.no_grad():
            loss = None
            if closure is not None:
                with crypten.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                betas = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                beta1, beta2 = betas

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    # Threshold gradients to prevent gradient explosion
                    if self.square_threshold is not None:
                        d_p = p.grad.mul(p.grad.square().lt(self.square_threshold))
                    else:
                        d_p = p.grad

                    if weight_decay != 0:
                        d_p = d_p.add(p.mul(weight_decay))

                    # Initialize state for running averages
                    param_state = self.state[id(p)]
                    if "m" not in param_state:
                        param_state["m"] = crypten.cryptensor(torch.zeros(p.shape))  # First moment estimate
                    if "v" not in param_state:
                        param_state["v"] = crypten.cryptensor(torch.zeros(p.shape))  # Second moment estimate
                    if "step" not in param_state:
                        param_state["step"] = torch.zeros(p.shape)  # Time step

                    m, v = param_state["m"], param_state["v"]
                    step = param_state["step"]

                    # Update moment estimates
                    m.mul_(beta1).add_(d_p.mul(1 - beta1))  # First moment estimate
                    v.mul_(beta2).add_(d_p.square().mul(1 - beta2))  # Second moment estimate

                    # Bias correction
                    bias_correction1 = 1 - beta1**(step + 1)
                    bias_correction2 = 1 - beta2**(step + 1)
                    m_hat = m / bias_correction1  # Corrected first moment
                    v_hat = v / bias_correction2  # Corrected second moment

                    # Update parameters
                    p.sub_(lr * m_hat / (v_hat.sqrt() + eps))

                    # Update time step
                    step.add_(1)

            return loss
