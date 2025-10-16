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
        """Forward function for MSE Loss.

        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.
            weight (torch.Tensor, optional): Element-wise weight of loss. Default is None.
            avg_factor (int, optional): Average factor for loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the loss. If None, use the class-level reduction.

        Returns:
            torch.Tensor: The calculated MSE loss.
        """

        # 计算均方误差 (MSE)
        mse_loss = ((target - pred) ** 2).sum()

        # 应用reduction方式
        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mse_loss = mse_loss / pred.numel()  # 平均化
        elif reduction == 'sum':
            pass  # 保持求和
        elif reduction == 'none':
            mse_loss = mse_loss.view(-1)  # 保持每个元素的损失

        # 如果提供了weight, 对损失进行加权
        if weight is not None:
            weight = weight.float()
            mse_loss = mse_loss * weight

        # 加权后应用损失权重
        mse_loss = self.loss_weight * mse_loss

        # 如果提供了avg_factor，进行平均
        if avg_factor is not None:
            mse_loss = mse_loss / avg_factor

        return mse_loss

@MODELS.register_module()
class MAPELoss(nn.Module):
    """平均绝对百分百误差损失函数。

    参数：
        reduction (str): 用于简化损失的方法。
            可选值为 "none"、"mean" 和 "sum"。默认为 'mean'。
        loss_weight (float): 损失的权重。默认为 1.0。
    """

    def __init__(self, reduction='sum', loss_weight=1.0):
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """前向计算函数，用于计算 MAPE 损失。

        参数：
            pred (torch.Tensor): 预测值。
            target (torch.Tensor): 真实值。
            weight (torch.Tensor, optional): 损失的元素级权重。默认为 None。
            avg_factor (int, optional): 损失的平均因子。默认为 None。
            reduction_override (str, optional): 用于简化损失的方法。如果为 None，则使用类级别的 reduction。

        返回：
            torch.Tensor: 计算得到的 MAPE 损失。
        """

        # 计算平均绝对误差 (MAPE)
        mape_loss = torch.abs((target - pred)/(target)).sum()

        # 应用 reduction 方式
        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mape_loss = mape_loss / pred.numel()  # 平均化
        elif reduction == 'sum':
            pass  # 保持求和
        elif reduction == 'none':
            mape_loss = mape_loss.view(-1)  # 保持每个元素的损失

        # 如果提供了 weight，对损失进行加权
        if weight is not None:
            weight = weight.float()
            mape_loss = mape_loss * weight

        # 加权后应用损失权重
        mape_loss = self.loss_weight * mape_loss

        # 如果提供了 avg_factor，进行平均
        if avg_factor is not None:
            mape_loss = mape_loss / avg_factor

        return mape_loss

@MODELS.register_module()
class SSELoss(nn.Module):
    """和方差损失函数。

    参数：
        reduction (str): 用于简化损失的方法。
            可选值为 "none"、"mean" 和 "sum"。默认为 'mean'。
        loss_weight (float): 损失的权重。默认为 1.0。
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """前向计算函数，用于计算 MAE 损失。

        参数：
            pred (torch.Tensor): 预测值。
            target (torch.Tensor): 真实值。
            weight (torch.Tensor, optional): 损失的元素级权重。默认为 None。
            avg_factor (int, optional): 损失的平均因子。默认为 None。
            reduction_override (str, optional): 用于简化损失的方法。如果为 None，则使用类级别的 reduction。

        返回：
            torch.Tensor: 计算得到的 SSE 损失。
        """

        # 计算和方差 (SSE)
        sse_loss = torch.abs((target - pred)*(target - pred)).sum()

        # 应用 reduction 方式
        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            sse_loss = sse_loss / pred.numel()  # 平均化
        elif reduction == 'sum':
            pass  # 保持求和
        elif reduction == 'none':
            sse_loss = sse_loss.view(-1)  # 保持每个元素的损失

        # 如果提供了 weight，对损失进行加权
        if weight is not None:
            weight = weight.float()
            sse_loss = sse_loss * weight

        # 加权后应用损失权重
        sse_loss = self.loss_weight * sse_loss

        # 如果提供了 avg_factor，进行平均
        if avg_factor is not None:
            sse_loss = sse_loss / avg_factor

        return sse_loss

@MODELS.register_module()
class MAELoss(nn.Module):
    """平均绝对误差损失函数。

    参数：
        reduction (str): 用于简化损失的方法。
            可选值为 "none"、"mean" 和 "sum"。默认为 'mean'。
        loss_weight (float): 损失的权重。默认为 1.0。
    """

    def __init__(self, reduction='sum', loss_weight=1.0):
        super(MAELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """前向计算函数，用于计算 MAE 损失。

        参数：
            pred (torch.Tensor): 预测值。
            target (torch.Tensor): 真实值。
            weight (torch.Tensor, optional): 损失的元素级权重。默认为 None。
            avg_factor (int, optional): 损失的平均因子。默认为 None。
            reduction_override (str, optional): 用于简化损失的方法。如果为 None，则使用类级别的 reduction。

        返回：
            torch.Tensor: 计算得到的 MAE 损失。
        """

        # 计算平均绝对误差 (MAE)
        mae_loss = torch.abs(target - pred).sum()

        # 应用 reduction 方式
        if reduction_override:
            reduction = reduction_override
        else:
            reduction = self.reduction

        if reduction == 'mean':
            mae_loss = mae_loss / pred.numel()  # 平均化
        elif reduction == 'sum':
            pass  # 保持求和
        elif reduction == 'none':
            mae_loss = mae_loss.view(-1)  # 保持每个元素的损失

        # 如果提供了 weight，对损失进行加权
        if weight is not None:
            weight = weight.float()
            mae_loss = mae_loss * weight

        # 加权后应用损失权重
        mae_loss = self.loss_weight * mae_loss

        # 如果提供了 avg_factor，进行平均
        if avg_factor is not None:
            mae_loss = mae_loss / avg_factor

        return mae_loss

@MODELS.register_module()
class HuberMSELoss(nn.Module):
    """
    Huber Loss：当误差较小时等效于 MSE，较大误差时切换为 MAE，从而减轻异常值的影响。

    参数：
        delta (float): 阈值。当 |error| <= delta 时采用 MSE，否则采用 MAE。
        reduction (str): 简化损失的方式，可选 "none"、"mean" 和 "sum"，默认为 "mean"。
        loss_weight (float): 损失权重，默认为 1.0。
    """

    def __init__(self, delta=1.0, reduction='mean', loss_weight=1.0):
        super(HuberMSELoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        error = pred - target
        abs_error = torch.abs(error)
        # 当 |error| <= delta 时，使用 0.5*error^2；否则使用 delta*(|error|-0.5*delta)
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
        # 'none' 下保持每个元素的损失不变

        loss = self.loss_weight * loss
        return loss


@MODELS.register_module()
class SmoothL1MSELoss(nn.Module):
    """
    Smooth L1 Loss：类似于 Huber Loss，但可以手动设置阈值，使得误差较小时接近 MSE，
    较大误差时接近 L1 Loss（绝对误差）。

    参数：
        beta (float): 阈值参数。当 |error| < beta 时使用 MSE，否则使用 L1 损失。默认为 1.0。
        reduction (str): 简化损失的方式，可选 "none"、"mean" 和 "sum"，默认为 "mean"。
        loss_weight (float): 损失权重，默认为 1.0。
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1MSELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        error = pred - target
        abs_error = torch.abs(error)
        # 当 |error| < beta 时，使用 0.5*(error^2)/beta；否则使用 |error| - 0.5*beta
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
    """
    MAE + 物理约束损失：在 MAE 损失之外加入物理约束损失，
    使得回归结果符合激光散斑粗糙度测量的物理规律。

    此处以预测值应落在 [min_val, max_val] 内为例：
      当预测值低于 min_val 或高于 max_val 时，将产生额外的损失。

    参数：
        min_val (float): 预测值的最小物理合理值，默认为 0.0。
        max_val (float): 预测值的最大物理合理值，默认为 1.0。
        physical_weight (float): 物理约束损失的权重，默认为 1.0。
        reduction (str): 简化损失的方式，可选 "none"、"mean" 和 "sum"，默认为 "mean"。
        loss_weight (float): 总损失权重，默认为 1.0。
    """

    def __init__(self, min_val=0.0, max_val=1.0, physical_weight=1.0, reduction='mean', loss_weight=1.0):
        super(MAEPhysicalLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.physical_weight = physical_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        # 使用 'none' 模式，由我们手动处理 reduction
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        # 计算 MAE 损失
        mae = self.mae_loss(pred, target)
        # 物理约束：当预测值低于 min_val 或高于 max_val 时施加惩罚
        physical_loss = torch.relu(self.min_val - pred) + torch.relu(pred - self.max_val)
        # 总损失为 MAE 损失加上物理约束损失
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
    """
    自定义加权损失：
    对不同误差数值范围进行不同加权，公式为：

      L = w1 * L_MSE (|error| < T1) + w2 * L_MAE (T1 ≤ |error| < T2) + w3 * L_Huber (|error| ≥ T2)

    参数：
        T1 (float): 第一段阈值，误差小于 T1 时采用 MSE 损失，默认为 1.0。
        T2 (float): 第二段阈值，误差大于等于 T2 时采用 Huber 损失，默认为 2.0。
        w1 (float): 第一段的权重，默认为 1.0。
        w2 (float): 第二段的权重，默认为 1.0。
        w3 (float): 第三段的权重，默认为 1.0。
        reduction (str): 简化损失的方式，可选 "none"、"mean" 和 "sum"，默认为 "mean"。
        loss_weight (float): 总损失权重，默认为 1.0。
    """

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
        # 第一段：误差 < T1，使用 MSE 损失（平方误差）
        loss_mse = error ** 2
        # 第二段：T1 ≤ 误差 < T2，使用 MAE 损失（绝对误差）
        loss_mae = abs_error
        # 第三段：误差 ≥ T2，使用 Huber 损失（以 T2 为阈值）
        loss_huber = torch.where(abs_error <= self.T2,
                                 0.5 * error ** 2,
                                 self.T2 * (abs_error - 0.5 * self.T2))

        # 根据误差范围创建掩码
        mask1 = (abs_error < self.T1).float()
        mask2 = ((abs_error >= self.T1) & (abs_error < self.T2)).float()
        mask3 = (abs_error >= self.T2).float()

        # 分段加权
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
    自定义动态权重损失：
    损失由三部分组成：
      - MAE：平均绝对误差；
      - MAPE：平均绝对百分比误差；
      - GLCM相关性损失：基于预测和真实相关性的绝对误差。

    总损失计算公式为：
      L = w1 * MAE + w2 * MAPE

    权重根据当前 epoch 的 mape 动态更新：
      - 当 mape > threshold 时：w1=1.0, w2=0.0, y_weight=0.0；
      - 当 mape <= threshold 时：w1=0.2, w2=0.8, y_weight=3.0。

    参数：
        threshold (float): MAPE 阈值，默认 20.0。
        w1_high (float): mape 大于阈值时 MAE 部分的权重，默认为 1.0。
        w2_high (float): mape 大于阈值时 MAPE 部分的权重，默认为 0.0。
        y_weight_high (float): mape 大于阈值时 GLCM 相关性部分的权重，默认为 0.0。
        w1_low (float): mape 小于等于阈值时 MAE 部分的权重，默认为 0.2。
        w2_low (float): mape 小于等于阈值时 MAPE 部分的权重，默认为 0.8。
        y_weight_low (float): mape 小于等于阈值时 GLCM 相关性部分的权重，默认为 3.0。
        loss_weight (float): 总损失缩放因子，默认为 1.0。
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

        # 初始权重设定为 mape 较高时的权重
        self.w1 = w1_high
        self.w2 = w2_high

        # 存储三种状态下的权重
        self.w1_high = w1_high
        self.w2_high = w2_high
        self.w1_low = w1_low
        self.w2_low = w2_low
        self.w1_lower = w1_lower
        self.w2_lower = w2_lower

        self.loss_weight = loss_weight

    def update_weights(self, current_mape):
        """
        根据当前 epoch 的 mape 更新内部权重：
          当 current_mape > threshold 时，采用高权重配置；
          否则采用低权重配置。
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
        """
        计算动态加权损失：
          - MAE：预测与真实值的平均绝对误差；
          - MAPE：预测与真实值的平均绝对百分比误差（百分比表示）；

        返回总损失和当前 batch 的 mape 值。
        """
        # 计算 MAE 和 MAPE（均值已经计算）
        error = pred - target
        mae = torch.mean(torch.abs(error))
        mape = torch.mean(torch.abs(error / (target + 1e-8))) * 100

        # 总损失加权求和
        total_loss = self.w1 * mae + self.w2 * mape

        # 应用总体损失缩放因子
        total_loss = self.loss_weight * total_loss

        return total_loss, mape

