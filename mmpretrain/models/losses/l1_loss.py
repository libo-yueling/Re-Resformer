import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from .utils import weight_reduce_loss


@MODELS.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none", "mean", and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        r"""Compute L1 loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, *).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, *), N or (N, 1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, *). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean", and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        pred = pred.to(target.device)
        # 计算L1损失
        loss = torch.abs(pred - target)

        # 应用权重
        if weight is not None:
            loss = loss * weight

        # 进行损失的规约
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return self.loss_weight * loss
