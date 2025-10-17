import torch
import torch.nn as nn
from mmpretrain.registry import MODELS


@MODELS.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error Loss.

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='sum', loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        mse_loss = ((target - pred) ** 2).sum()

        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mse_loss = mse_loss / pred.numel() 
        elif reduction == 'sum':
            pass  
        elif reduction == 'none':
            mse_loss = mse_loss.view(-1)  

        if weight is not None:
            weight = weight.float()
            mse_loss = mse_loss * weight

        mse_loss = self.loss_weight * mse_loss

        if avg_factor is not None:
            mse_loss = mse_loss / avg_factor

        return mse_loss

@MODELS.register_module()
class MAPELoss(nn.Module):
    def __init__(self, reduction='sum', loss_weight=1.0):
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward computation function to compute the MAPE loss.

        Parameters:
            pred (torch.Tensor): Predictions.
            target (.Tensor): Target values.
            weight (torch.Tensor, optional): Element-wise weights for the loss. Default is None.
            avg_factor (int,): Factor to average the loss. Default is None.
            reduction_override (str, optional): Method to reduce the loss. If None, class-level reduction is.

        Returns:
            torch.Tensor: Calculated MAPE loss.
        """

        mape_loss = torch.abs((target - pred)/(target)).sum()

        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mape_loss = mape_loss / pred.numel() 
        elif reduction == 'sum':
            pass 
        elif reduction == 'none':
            mape_loss = mape_loss.view(-1)  

        if weight is not None:
            weight = weight.float()
            mape_loss = mape_loss * weight

        mape_loss = self.loss_weight * mape_loss

        if avg_factor is not None:
            mape_loss = mape_loss / avg_factor

        return mape_loss

@MODELS.register_module()
class SSELoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):

        sse_loss = torch.abs((target - pred)*(target - pred)).sum()

        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            sse_loss = sse_loss / pred.numel() 
        elif reduction == 'sum':
            pass 
        elif reduction == 'none':
            sse_loss = sse_loss.view(-1)  

        if weight is not None:
            weight = weight.float()
            sse_loss = sse_loss * weight

        sse_loss = self.loss_weight * sse_loss

        if avg_factor is not None:
            sse_loss = sse_loss / avg_factor

        return sse_loss

@MODELS.register_module()
class MAELoss(nn.Module):
    def __init__(self, reduction='sum', loss_weight=1.0):
        super(MAELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        mae_loss = torch.abs(target - pred).sum()

        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mae_loss = mae_loss / pred.numel()  
        elif reduction == 'sum':
            pass  
        elif reduction == 'none':
            mae_loss = mae_loss.view(-1)  

        if weight is not None:
            weight = weight.float()
            mae_loss = mae_loss * weight

        mae_loss = self.loss_weight * mae_loss

        if avg_factor is not None:
            mae_loss = mae_loss / avg_factor

        return mae_loss

@MODELS.register_module()
class HuberMSELoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean', loss_weight=1.0):
        super(HuberMSELoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        error = pred - target
        abs_error = torch.abs(error)
        loss = torch.where(abs_error <= self.delta,
                           0.5 * error ** 2,
                           self.delta * (abs_error - 0.5 * self.delta))

        if weight is not None:
            loss = loss * weight.float()

        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction == 'mean':
            loss = loss.sum() / pred.numel() if avg_factor is None else loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        loss = self.loss_weight * loss
        return loss


@MODELS.register_module()
class SmoothL1MSELoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1MSELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        error = pred - target
        abs_error = torch.abs(error)
        loss = torch.where(abs_error < self.beta,
                           0.5 * (error ** 2) / self.beta,
                           abs_error - 0.5 * self.beta)

        if weight is not None:
            loss = loss * weight.float()

        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction == 'mean':
            loss = loss.sum() / pred.numel() if avg_factor is None else loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        loss = self.loss_weight * loss
        return loss


@MODELS.register_module()
class MAEPhysicalLoss(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0, physical_weight=1.0, reduction='mean', loss_weight=1.0):
        super(MAEPhysicalLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.physical_weight = physical_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        physical_loss = torch.relu(self.min_val - pred) + torch.relu(pred - self.max_val)
        loss = mae + self.physical_weight * physical_loss

        if weight is not None:
            loss = loss * weight.float()

        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction == 'mean':
            loss = loss.sum() / pred.numel() if avg_factor is None else loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        loss = self.loss_weight * loss
        return loss

@MODELS.register_module()
class CustomWeightedLoss(nn.Module):
    def __init__(self, T1=1.0, T2=2.0, w1=1.0, w2=1.0, w3=1.0, reduction='mean', loss_weight=1.0):
        super(CustomWeightedLoss, self).__init__()
        self.T1 = T1
        self.T2 = T2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        error = pred - target
        abs_error = torch.abs(error)
        loss_mse = error ** 2
        loss_mae = abs_error
        loss_huber = torch.where(abs_error <= self.T2,
                                 0.5 * error ** 2,
                                 self.T2 * (abs_error - 0.5 * self.T2))

        mask1 = (abs_error < self.T1).float()
        mask2 = ((abs_error >= self.T1) & (abs_error < self.T2)).float()
        mask3 = (abs_error >= self.T2).float()

        loss = self.w1 * loss_mse * mask1 + self.w2 * loss_mae * mask2 + self.w3 * loss_huber * mask3

        if weight is not None:
            loss = loss * weight.float()

        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction == 'mean':
            loss = loss.sum() / pred.numel() if avg_factor is None else loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        loss = self.loss_weight * loss
        return loss


@MODELS.register_module()
class DynamicLoss(nn.Module):
    """
    Custom dynamic weight loss:
    The loss consists of three parts:
      - MAE: Mean Absolute Error;
       MAPE: Mean Absolute Percentage Error;
      - GLCM correlation loss: Absolute error based on the correlation of prediction and truth.

    Total is calculated as:
      L = w1 * MAE   w2 * MAPE

    Weights are dynamically updated according to the mape of the current epoch
      - When mape > threshold: w1=1.0, w2=0.0, y_weight=0.0;
      - When mape threshold: w1=0.2, w2=0.8, y_weight=3.0.

    Parameters:
        threshold (float): MAP threshold, default 20.0.
        w1_high (float): Weight of MAE part when mape is greater than threshold, default 1.0
        w2_high (float): Weight of MAPE part when mape is greater than threshold, default 0.0.
        y_weight_high (): Weight of GLCM correlation part when mape is greater than threshold, default 0.0.
        w1_low (float): Weight of MAE part whenape is less than or equal to threshold, default 0.2.
        w2_low (float): Weight of MAPE part when mape is less than or to threshold, default 0.8.
        y_weight_low (float): Weight of GLCM correlation part when mape is less than or equal to threshold default 3.0.
        loss_weight (float): Total loss scaling factor, default 1.0.
    """

    def __init__(self,
                 threshold1=20.0,threshold2=8.0,
                 w1_high=1.0, w2_high=0.0,
                 w1_low=0.2, w2_low=0.8,
                 w1_lower=0.0, w2_lower=1.0,
                 loss_weight=1.0):
        super(DynamicLoss, self).__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.w1 = w1_high
        self.w2 = w2_high

        self.w1_high = w1_high
        self.w2_high = w2_high
        self.w1_low = w1_low
        self.w2_low = w2_low
        self.w1_lower = w1_lower
        self.w2_lower = w2_lower

        self.loss_weight = loss_weight

    def update_weights(self, current_mape):
        """
        Update the internal weights based on the mape of the current epoch:
          Use the high weight configuration when current_mape > threshold;
          Other, use the low weight configuration.
        """
        if current_mape > self.threshold1:
            self.w1 = self.w1_high
            self.w2 = self.w2_high
        if current_mape in[self.threshold2, self.threshold1]:
            self.w1 = self.w1_low
            self.w2 = self.w2_low
        else:
            self.w1 = self.w1_lower
            self.w2 = self.w2_lower

    def forward(self, pred, target):
        error = pred - target
        mae = torch.mean(torch.abs(error))
        mape = torch.mean(torch.abs(error / (target + 1e-8))) * 100
        total_loss = self.w1 * mae + self.w2 * mape
        total_loss = self.loss_weight * total_loss
        return total_loss, mape

