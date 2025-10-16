import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple, Union


# **定义 KAN 模块**
class KANLayer(nn.Module):
    """简单的 KAN（Kolmogorov-Arnold Networks）层。"""

    def __init__(self, in_features, out_features, hidden_units=1):
        super(KANLayer, self).__init__()
        self.hidden_units = hidden_units

        # KAN 的非线性变换部分
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(hidden_units, out_features)

    def forward(self, x):
        x = torch.sin(x)
        x = self.gelu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class MLPRegressor(nn.Module):
    """
    简单的两层 MLP，把输入特征 [B, C]（或 [B, C, 1, 1]）映射到 [B, 1]。
    - 首先把 [B, C, 1, 1] squeeze 成 [B, C]，然后过 (C -> hidden) + ReLU -> (hidden -> 1)。
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, out_features: int = 1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)   # [B, hidden_dim]
        x = self.relu(x)
        out = self.fc2(x) # [B, 1]
        return out

class LinearRegressor(nn.Module):
    """
    线性回归模块，把输入特征 [B, C]（或 [B, C, 1, 1]）直接映射到 [B, 1]。
    - 如果输入是 [B, C, 1, 1]：会先 squeeze 成 [B, C]，再通过线性层得到 [B, 1]。
    - 如果输入是 [B, C]：直接过线性层得到 [B, 1]。
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super(LinearRegressor, self).__init__()
        # 一个全连接层：C -> 1
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状要么是 [B, C, 1, 1]，要么是 [B, C]
        Returns:
            torch.Tensor: [B, 1]
        """
        # 如果是 [B, C, 1, 1]，先 squeeze 成 [B, C]
        if x.ndim == 4 and x.shape[2:] == (1, 1):
            x = x.view(x.size(0), x.size(1))

        # 此时 x 应该是 [B, C]
        out = self.fc(x)  # [B, 1]
        return out

@MODELS.register_module()
class KaRHead(BaseModule):
    """Head for regression (with linear regression for roughness)."""

    def __init__(self,
                 depth,
                 in_channels: int,
                 roughness_loss: dict = dict(type='DynamicLoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 init_cfg: Optional[dict] = None):
        super(KaRHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roughness_loss_module = MODELS.build(roughness_loss)
        if depth in [18, 34]:
            features = 512
        elif depth in [50, 101, 152]:
            features = 2048
        # Initialize regressor for roughness regression
        self.regressor_after_pooling = KANLayer(features, regressor_out_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> torch.Tensor:
    # 使用最后一个阶段的特征
        last_feat = feats[-1]                  # Tensor 形状如 [B, C, H, W]
        pooled = last_feat.view(last_feat.size(0), -1)  # 展平成 [B, C*H*W]

        # 用 KANLayer 回归粗糙度
        roughness_value = self.regressor_after_pooling(pooled)
        return roughness_value


    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the roughness value."""
        if data_samples is None:
            raise ValueError("data_samples should not be None.")

        roughness_value = self(feats)

        # Unpack data samples safely, handling None values
        targets_roughness = []

        for ds in data_samples:
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)

        # Ensure targets are not empty
        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")

        targets_roughness = torch.cat(targets_roughness)

        # Ensure targets_roughness is on the same device as the model's output
        device = roughness_value.device
        targets_roughness = targets_roughness.to(device)

        # Calculate roughness loss
        losses = dict()
        loss_roughness,mape = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
        losses['loss_roughness'] = loss_roughness
        losses['mape'] = mape

        # 计算总的粗糙度损失
        losses['total_loss_roughness'] = loss_roughness  # 直接使用计算的损失

        return losses
    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[
        DataSample]:
        """Inference without augmentation."""
        roughness_value = self(feats)

        if data_samples is None:
            data_samples = [None for _ in range(roughness_value.size(0))]

        predictions = self._get_predictions(roughness_value, data_samples)

        # # Print real labels and predictions for debugging
        # for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
        #     if data_sample is not None:
        #         print(f"Sample {i + 1}:")
        #         print(f"Real Roughness: {data_sample.gt_roughness}")
        #         print(f"Predicted Roughness: {prediction.pred_roughness}")
        #         print()

        return predictions

    def _get_predictions(self, roughness_value: torch.Tensor,
                         data_samples: List[Optional[DataSample]]) -> List[DataSample]:
        """Generate prediction results for each sample."""
        predictions = []

        for i, (roughness, data_sample) in enumerate(zip(roughness_value, data_samples)):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.pred_roughness = roughness  # No softmax here for roughness

            predictions.append(data_sample)

        return predictions

@MODELS.register_module()
class LinearRoughnessHead(BaseModule):
    """Head for regression (with linear regression for roughness)."""

    def __init__(self,
                 depth,
                 in_channels: int,
                 roughness_loss: dict = dict(type='DynamicLoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 init_cfg: Optional[dict] = None):
        super(LinearRoughnessHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        if depth in [18, 34]:
            features = 512
        elif depth in [50, 101, 152]:
            features = 2048
        self.roughness_loss_module = MODELS.build(roughness_loss)

        # Initialize regressor for roughness regression
        self.regressor_after_pooling = LinearRegressor(features, regressor_out_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> torch.Tensor:
    # 使用最后一个阶段的特征
        last_feat = feats[-1]                  # Tensor 形状如 [B, C, H, W]
        pooled = last_feat.view(last_feat.size(0), -1)  # 展平成 [B, C*H*W]

        # 用 LinearRegressor 回归粗糙度
        roughness_value = self.regressor_after_pooling(pooled)
        return roughness_value


    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the roughness value."""
        if data_samples is None:
            raise ValueError("data_samples should not be None.")

        roughness_value = self(feats)

        # Unpack data samples safely, handling None values
        targets_roughness = []

        for ds in data_samples:
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)

        # Ensure targets are not empty
        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")

        targets_roughness = torch.cat(targets_roughness)

        # Ensure targets_roughness is on the same device as the model's output
        device = roughness_value.device
        targets_roughness = targets_roughness.to(device)

        # Calculate roughness loss
        losses = dict()
        loss_roughness,mape = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
        losses['loss_roughness'] = loss_roughness
        losses['mape'] = mape

        # 计算总的粗糙度损失
        losses['total_loss_roughness'] = loss_roughness  # 直接使用计算的损失

        return losses
    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[
        DataSample]:
        """Inference without augmentation."""
        roughness_value = self(feats)

        if data_samples is None:
            data_samples = [None for _ in range(roughness_value.size(0))]

        predictions = self._get_predictions(roughness_value, data_samples)

        # # Print real labels and predictions for debugging
        # for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
        #     if data_sample is not None:
        #         print(f"Sample {i + 1}:")
        #         print(f"Real Roughness: {data_sample.gt_roughness}")
        #         print(f"Predicted Roughness: {prediction.pred_roughness}")
        #         print()

        return predictions

    def _get_predictions(self, roughness_value: torch.Tensor,
                         data_samples: List[Optional[DataSample]]) -> List[DataSample]:
        """Generate prediction results for each sample."""
        predictions = []

        for i, (roughness, data_sample) in enumerate(zip(roughness_value, data_samples)):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.pred_roughness = roughness  # No softmax here for roughness

            predictions.append(data_sample)

        return predictions

@MODELS.register_module()
class MLPRoughnessHead(BaseModule):
    """Head for regression (with linear regression for roughness)."""

    def __init__(self,
                 depth,
                 in_channels: int,
                 roughness_loss: dict = dict(type='DynamicLoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 init_cfg: Optional[dict] = None):
        super(MLPRoughnessHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roughness_loss_module = MODELS.build(roughness_loss)
        if depth in [18, 34]:
            features = 512
        elif depth in [50, 101, 152]:
            features = 2048
        # Initialize regressor for roughness regression
        self.regressor_after_pooling = MLPRegressor(features, regressor_out_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> torch.Tensor:
    # 使用最后一个阶段的特征
        last_feat = feats[-1]                  # Tensor 形状如 [B, C, H, W]
        pooled = last_feat.view(last_feat.size(0), -1)  # 展平成 [B, C*H*W]

        # 用 KANLayer 回归粗糙度
        roughness_value = self.regressor_after_pooling(pooled)
        return roughness_value


    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the roughness value."""
        if data_samples is None:
            raise ValueError("data_samples should not be None.")

        roughness_value = self(feats)

        # Unpack data samples safely, handling None values
        targets_roughness = []

        for ds in data_samples:
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)

        # Ensure targets are not empty
        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")

        targets_roughness = torch.cat(targets_roughness)

        # Ensure targets_roughness is on the same device as the model's output
        device = roughness_value.device
        targets_roughness = targets_roughness.to(device)

        # Calculate roughness loss
        losses = dict()
        loss_roughness,mape = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
        losses['loss_roughness'] = loss_roughness
        losses['mape'] = mape

        # 计算总的粗糙度损失
        losses['total_loss_roughness'] = loss_roughness  # 直接使用计算的损失

        return losses
    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[
        DataSample]:
        """Inference without augmentation."""
        roughness_value = self(feats)

        if data_samples is None:
            data_samples = [None for _ in range(roughness_value.size(0))]

        predictions = self._get_predictions(roughness_value, data_samples)

        # # Print real labels and predictions for debugging
        # for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
        #     if data_sample is not None:
        #         print(f"Sample {i + 1}:")
        #         print(f"Real Roughness: {data_sample.gt_roughness}")
        #         print(f"Predicted Roughness: {prediction.pred_roughness}")
        #         print()

        return predictions

    def _get_predictions(self, roughness_value: torch.Tensor,
                         data_samples: List[Optional[DataSample]]) -> List[DataSample]:
        """Generate prediction results for each sample."""
        predictions = []

        for i, (roughness, data_sample) in enumerate(zip(roughness_value, data_samples)):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.pred_roughness = roughness  # No softmax here for roughness

            predictions.append(data_sample)

        return predictions