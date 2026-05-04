from __future__ import annotations

import math
from typing import Iterable

import torch


def _orthogonalize_update(g: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    shape = g.shape
    x = g.reshape(g.shape[0], -1).float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()

    norm = x.norm()
    if norm == 0:
        return torch.zeros_like(g)
    x = x / (norm + eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.t()
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.t()
    return x.reshape(shape).to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for matrix-like parameters with AdamW fallback groups.

    Param groups with ``use_muon=False`` use AdamW. This keeps biases,
    normalization weights, and small output heads off Muon.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        adamw_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if group["use_muon"]:
                    self._muon_step(p, group)
                else:
                    self._adamw_step(p, group)

        return loss

    def _muon_step(self, p: torch.Tensor, group: dict):
        grad = p.grad
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)

        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(grad)
        update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
        update = _orthogonalize_update(update, group["ns_steps"], group["adamw_eps"])

        flat = p.reshape(p.shape[0], -1)
        update.mul_(math.sqrt(max(1.0, flat.size(0) / flat.size(1))))

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        p.add_(update, alpha=-group["lr"])

    def _adamw_step(self, p: torch.Tensor, group: dict):
        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["adamw_betas"]
        state["step"] += 1

        lr = group["adamw_lr"]
        if group["weight_decay"]:
            p.mul_(1.0 - lr * group["weight_decay"])

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["adamw_eps"]), value=-step_size)


def make_muon_optimizer(
    module: torch.nn.Module,
    lr: float,
    adamw_lr: float,
    weight_decay: float,
    momentum: float,
    ns_steps: int,
) -> Muon:
    muon_params = []
    adamw_params = []
    final_heads = ("movement_head.2", "button_head.2", "value_head.2")

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        flat = p.reshape(p.shape[0], -1) if p.ndim >= 2 else None
        is_hidden_matrix = p.ndim >= 2 and flat is not None and flat.size(0) >= 16
        if is_hidden_matrix and not name.startswith(final_heads):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = []
    if muon_params:
        param_groups.append({"params": muon_params, "use_muon": True})
    if adamw_params:
        param_groups.append({
            "params": adamw_params,
            "use_muon": False,
            "weight_decay": 0.0,
        })

    return Muon(
        param_groups,
        lr=lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        momentum=momentum,
        ns_steps=ns_steps,
    )
