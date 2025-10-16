import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple
import numpy as np
from skimage import  feature
from sklearn import preprocessing
import cv2


# **定义 KAN 模块**
class KANLayer(nn.Module):
    """简单的 KAN（Kolmogorov-Arnold Networks）层。"""

    def __init__(self, in_features, out_features, hidden_units=1):
        super(KANLayer, self).__init__()
        self.hidden_units = hidden_units

        # KAN 的非线性变换部分
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.gelu = nn.GELU()  # 定义 GELU 激活函数
        self.output_layer = nn.Linear(hidden_units, out_features)

    def forward(self, x):
        x = self.gelu(self.hidden_layer(x))  # 采用 GELU 激活函数
        x = self.output_layer(x)
        return x

@MODELS.register_module()
class KaRCHead(BaseModule):
    """Head for regression (with linear regression for roughness) and GLCM feature extraction."""

    def __init__(self,
                 in_channels: int,
                 roughness_loss: dict = dict(type='MAELoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 init_cfg: Optional[dict] = None):
        super(KaRCHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.roughness_loss_module = MODELS.build(roughness_loss)

        # 初始化用于粗糙度回归的全连接层，这里以 KANLayer 为示例
        self.regressor_after_pooling = KANLayer(960, regressor_out_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """获取用于 head 的最后一层特征。"""
        return feats[-1]

    def compute_glcm_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算输入图像的 GLCM 特征。
        参数:
          image (torch.Tensor): 单通道图像张量，形状应为 (H, W)
        返回:
          torch.Tensor: 包含 GLCM 各项特征的一维张量
        """
        # 将 tensor 转为 numpy 数组（注意：这里假设 image 在 CPU 上）
        img = image.detach().cpu().numpy()
        img = img.squeeze(0)
        # 对图像进行归一化并映射到 [0, 63] 整数区间
        S = preprocessing.MinMaxScaler((0, 63)).fit_transform(img).astype(int)
        # 计算灰度共生矩阵，使用多种距离和角度
        glcm = feature.graycomatrix(S,
                                    distances=[1, 2, 3],
                                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                    levels=64,
                                    symmetric=False,
                                    normed=True)
        # 计算相关性属性
        correlation = feature.graycoprops(glcm, 'correlation')
        # 取所有角度和距离的均值
        avg_correlation = np.mean(correlation)

        # 将各项特征合并成一个 tensor 返回
        cor_pred= torch.tensor([
            avg_correlation
        ], dtype=torch.float32)

        return cor_pred

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # 粗糙度回归部分
        pooled_features = torch.cat([feat.view(feat.size(0), -1) for feat in feats], dim=1)
        roughness_value = self.regressor_after_pooling(pooled_features)

        return roughness_value

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        if data_samples is None:
            raise ValueError("data_samples should not be None.")
        # 获取粗糙度回归输出
        roughness_value = self(feats)
        # 计算 GLCM 相关性特征：使用 data_samples 中存储的原始图像
        if len(data_samples) > 0 and hasattr(data_samples[0], 'ori_img') and data_samples[0].ori_img is not None:
            cor_pred = self.compute_glcm_features(data_samples[0].ori_img)
        else:
            # 回退方案：例如从特征图中计算（不推荐，因为这可能不符合物理约束）
            cor_pred = self.compute_glcm_features(feats[-1][0, 0, :, :])

        # 从 data_samples 中提取目标粗糙度和目标相关性（gt_correlation）
        targets_roughness = []
        targets_correlation = []
        for ds in data_samples:
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)
            if hasattr(ds, 'gt_correlation') and ds.gt_correlation is not None:
                # 确保 gt_correlation 是1D张量
                targets_correlation.append(ds.gt_correlation.view(1))

        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")
        if not targets_correlation:
            raise ValueError("No valid correlation values found in data samples.")

        targets_roughness = torch.cat(targets_roughness).to(roughness_value.device)
        targets_correlation = torch.cat(targets_correlation).to(roughness_value.device)

        # 调用动态损失模块（假设它接受粗糙度预测、目标、相关性预测和目标相关性作为输入）
        loss_roughness, mape, loss_correlation = self.roughness_loss_module(
            roughness_value, targets_roughness, cor_pred, targets_correlation, **kwargs
        )

        losses = dict()
        losses['loss_roughness'] = loss_roughness
        losses['mape'] = mape
        losses['loss_correlation'] = loss_correlation
        losses['total_loss_roughness'] = loss_roughness + loss_correlation

        return losses

    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[
        DataSample]:
        """推理，不做数据增强。"""
        roughness_value, _ = self(feats)
        if data_samples is None:
            data_samples = [None for _ in range(roughness_value.size(0))]
        predictions = self._get_predictions(roughness_value, data_samples)

        # 调试输出预测与真实标签
        for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
            if data_sample is not None:
                print(f"Sample {i + 1}:")
                print(f"Real Roughness: {data_sample.gt_roughness}")
                print(f"Predicted Roughness: {prediction.pred_roughness}")
                print()
        return predictions

    def _get_predictions(self, roughness_value: torch.Tensor, data_samples: List[Optional[DataSample]]) -> List[
        DataSample]:
        """为每个样本生成预测结果。"""
        predictions = []
        for i, (roughness, data_sample) in enumerate(zip(roughness_value, data_samples)):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.pred_roughness = roughness  # 直接使用回归结果，无 softmax 操作
            predictions.append(data_sample)
        return predictions

