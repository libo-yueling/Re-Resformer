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


@MODELS.register_module()
class KaRHead(BaseModule):
    """Head for regression (with linear regression for roughness)."""

    def __init__(self,
                 in_channels: int,
                 roughness_loss: dict = dict(type='MAELoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 init_cfg: Optional[dict] = None):
        super(KaRHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roughness_loss_module = MODELS.build(roughness_loss)

        # Initialize regressor for roughness regression
        self.regressor_after_pooling = KANLayer(3840, regressor_out_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> torch.Tensor:
        # 将所有层的特征进行拼接
        # for i in range(len(feats)):
        #     print(feats[i])
        #     print(feats[i].size())
        pooled_features_for_regression = torch.cat([feat.view(feat.size(0), -1) for feat in feats], dim=1)

        # 回归粗糙度
        roughness_value = self.regressor_after_pooling(pooled_features_for_regression)

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
        loss_roughness, mape = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
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

        # Print real labels and predictions for debugging
        for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
            if data_sample is not None:
                print(f"Sample {i + 1}:")
                print(f"Real Roughness: {data_sample.gt_roughness}")
                print(f"Predicted Roughness: {prediction.pred_roughness}")
                print()

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

