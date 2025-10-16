import torch
import torch.nn as nn
from mmpretrain.registry import MODELS


@MODELS.register_module()
class DDynamicLoss(nn.Module):
    """
    动态损失函数，包含 MAE、MAPE 和对比度约束损失。
    对比度约束部分通过软分段根据 cor_true（粗糙度）的不同区间调整权重。
    """

    def __init__(self,
                 threshold=20.0,
                 w1_high=1.0, w2_high=0.0, y_weight_high=0.0,
                 w1_low=0.2, w2_low=0.8, y_weight_low=3.0,
                 loss_weight=1.0):
        super(DDynamicLoss, self).__init__()
        self.threshold = threshold
        # 初始权重设定为 mape 较高时的权重
        self.w1 = w1_high
        self.w2 = w2_high
        self.y_weight = y_weight_high

        # 存储两种状态下的权重
        self.w1_high = w1_high
        self.w2_high = w2_high
        self.y_weight_high = y_weight_high
        self.w1_low = w1_low
        self.w2_low = w2_low
        self.y_weight_low = y_weight_low

        self.loss_weight = loss_weight

    def update_weights(self, current_mape):
        """
        根据当前 epoch 的 mape 更新权重。
        """
        if current_mape > self.threshold:
            self.w1 = self.w1_high
            self.w2 = self.w2_high
            self.y_weight = self.y_weight_high
        else:
            self.w1 = self.w1_low
            self.w2 = self.w2_low
            self.y_weight = self.y_weight_low

    def soft_segment_indicator(self, x, lower, upper, sharpness=50.0):
        """
        返回一个软分段指示值：
          当 x 落在 [lower, upper] 区间内时，输出接近 1，
          边界附近平滑过渡。
        参数:
          - x: 待判断的值（tensor）
          - lower: 区间下界
          - upper: 区间上界
          - sharpness: 控制平滑程度，值越大转变越陡峭
        """
        return torch.sigmoid(sharpness * (x - lower)) - torch.sigmoid(sharpness * (x - upper))

    def forward(self, pred, target, cor_pred, cor_true):
        """
        参数:
          - pred: 模型输出的回归预测
          - target: 对应的真实值
          - cor_pred: 模型输出的对比度特征（由网络 head 提取）
          - cor_true: 粗糙度的 ground truth，建议在 dataloader 中一起传入
        计算:
          - MAE：平均绝对误差
          - MAPE：平均绝对百分比误差
          - 对比度约束损失：先通过软分段获得针对 cor_true 的缩放系数，再计算加权绝对误差
        """
        # 计算 MAE 和 MAPE
        error = pred - target
        mae = torch.mean(torch.abs(error))
        mape = torch.mean(torch.abs(error / (target + 1e-8))) * 100

        # 软分段计算：针对 con_true（粗糙度）的不同范围进行平滑指示
        seg1 = self.soft_segment_indicator(cor_true, 0.648, 1.595)
        seg2 = self.soft_segment_indicator(cor_true, 1.914, 3.138)
        seg3 = self.soft_segment_indicator(cor_true, 3.451, 6.438)
        seg4 = self.soft_segment_indicator(cor_true, 6.438, 7.303)
        seg5 = self.soft_segment_indicator(cor_true, 7.303, 7.975)
        seg6 = self.soft_segment_indicator(cor_true, 8.290, 9.259)

        # 如果有不同区间对应的线性关系，可以在此定义每个区间的缩放因子
        # 例如这里简单将各区间的缩放因子都设为 1.0，可根据实际需求调整
        scale1, scale2, scale3, scale4, scale5, scale6 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

        # 综合各区间的缩放因子，注意各段可能有重叠或零权重部分
        scaling = seg1 * scale1 + seg2 * scale2 + seg3 * scale3 + seg4 * scale4 + seg5 * scale5 + seg6 * scale6

        # 计算对比度约束损失，使用软分段缩放调制
        cor_loss = self.y_weight * scaling * torch.abs(cor_pred - cor_true)

        # 总损失
        total_loss = self.w1 * mae + self.w2 * mape + cor_loss
        total_loss = self.loss_weight * total_loss

        return total_loss, mape , cor_loss
