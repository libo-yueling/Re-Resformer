import torch
from mmpretrain.registry import OPTIMIZERS
from torch.optim import Optimizer
from collections import defaultdict

@OPTIMIZERS.register_module()
class pv_Optimizer(Optimizer):
    """自定义优化器，基于Adam优化器的功能并确保参数值大于0."""

    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0001,
                 **kwargs):
        # 参数有效性检查
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"无效的betas值: {betas}")
        if eps < 0.0:
            raise ValueError(f"无效的eps值: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的weight_decay值: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # 初始化一阶矩和二阶矩
        self.state = defaultdict(dict)

        # 确保每个param_group包含step
        for group in self.param_groups:
            group.setdefault('step', 0)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单次优化步骤，确保所有参数值大于0."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 对每个参数组进行优化
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取当前参数的梯度
                grad = p.grad
                state = self.state[p]

                # 初始化一阶矩和二阶矩
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.contiguous_format)
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.contiguous_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # 更新一阶矩和二阶矩的指数加权平均
                exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                # 对二阶矩做偏置修正
                bias_correction1 = 1 - betas[0]**(group['step'] + 1)
                bias_correction2 = 1 - betas[1]**(group['step'] + 1)

                denom = exp_avg_sq.sqrt().add_(eps)  # 加上eps防止除零

                # 计算更新值
                step_size = lr / bias_correction1
                param_update = exp_avg / denom
                p.data.add_(-step_size, param_update)

                # 权重衰减（L2正则化）
                if weight_decay != 0:
                    p.data.add_(-lr * weight_decay, p.data)

                # 确保所有参数值大于0
                p.data = torch.clamp(p.data, min=0.0)

            # 更新 step 字段
            group['step'] += 1  # 这里更新每个参数组的step

        return loss
