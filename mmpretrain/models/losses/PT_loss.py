import torch
import torch.nn as nn
from mmpretrain.registry import MODELS

@MODELS.register_module()
class PTLoss(nn.Module):
    """Mean Squared Error Loss with regularization to enforce prediction bounds [0, 15].

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        penalty_weight (float): The weight of the penalty term. Defaults to 1.0.
        min_value (float): The minimum value of the prediction, defaults to 0.
        max_value (float): The maximum value of the prediction, defaults to 15.
    """

    def __init__(self, reduction='sum', loss_weight=1.0, penalty_weight=1.0, min_value=0.0, max_value=15.0):
        super(PTLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.penalty_weight = penalty_weight
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function for PT Loss with boundary constraint.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The ground truth values.
            weight (torch.Tensor, optional): Element-wise weight of loss. Default is None.
            avg_factor (int, optional): Average factor that is used to average the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the loss. If None, use the class-level reduction.

        Returns:
            torch.Tensor: The calculated loss with boundary penalty.
        """

        # Ensure predictions and targets don't contain NaN or Inf
        if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
            raise ValueError("Prediction contains NaN or Inf values.")
        if torch.any(torch.isnan(target)) or torch.any(torch.isinf(target)):
            raise ValueError("Target contains NaN or Inf values.")

        # Ensure the target labels are within valid range for cross_entropy
        n_classes = pred.size(1)  # Assuming pred has shape [batch_size, num_classes, ...]
        if torch.any(target < 0) or torch.any(target >= n_classes):
            raise ValueError(f"Target labels must be between 0 and {n_classes - 1}, but found out-of-range values.")

        # Ensure the shapes of pred and target match
        assert pred.shape[0] == target.shape[0], f"Shape mismatch: pred shape {pred.shape}, target shape {target.shape}"

        # Calculate the MSE loss
        mse_loss = ((target - pred) ** 2).sum()

        # Apply the reduction method
        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mse_loss = mse_loss / pred.numel()
        elif reduction == 'sum':
            pass  # keep the sum as is
        elif reduction == 'none':
            mse_loss = mse_loss.view(-1)  # retain the per-element loss

        # Apply the weight (if provided) and the loss_weight
        if weight is not None:
            weight = weight.float()
            mse_loss = mse_loss * weight

        # Apply loss weight
        mse_loss = self.loss_weight * mse_loss

        # Average by the avg_factor if provided
        if avg_factor is not None:
            mse_loss = mse_loss / avg_factor

        # Apply boundary penalty (if predictions go beyond [0, 15])
        penalty = 0.0
        if self.penalty_weight > 0.0:
            # Penalize values below min_value (0)
            penalty += torch.sum(torch.clamp(pred, max=self.min_value) - self.min_value)
            # Penalize values above max_value (15)
            penalty += torch.sum(torch.clamp(pred, min=self.max_value) - self.max_value)

        # Add penalty to loss
        mse_loss += self.penalty_weight * penalty

        return mse_loss
