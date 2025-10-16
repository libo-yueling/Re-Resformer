import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple, Union


@MODELS.register_module()
class MultiOutputHead(BaseModule):
    """Multi-Output head for classification and regression (with linear regression for roughness)."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 class_loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 roughness_loss: dict = dict(type='PTLoss', loss_weight=1.0),
                 regressor_out_channels: int = 1,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(MultiOutputHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.topk = topk
        self.class_loss_module = MODELS.build(class_loss)
        self.roughness_loss_module = MODELS.build(roughness_loss)
        self.cal_acc = cal_acc

        # Initialize classifier for classification
        self.classifier = nn.Linear(in_channels, num_classes)

        # Initialize regressor for roughness regression
        self.regressor_after_pooling = nn.Linear(4, regressor_out_channels)

        # Global Average Pooling layer (for reducing each feature map to a single value)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Used for each layer's feature

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 这里 feats 已经包含了 Backbone 输出的所有层的特征
        pooled_features = []

        # 对每一层的特征进行全局平均池化
        for feature in feats:
            pooled_feature = self.global_avg_pool(feature).view(feature.size(0), -1)
            pooled_features.append(pooled_feature)
        print(pooled_features)

        # 将所有特征拼接起来进行回归
        pooled_features_for_regression = torch.cat(pooled_features, dim=1)
        roughness_value = self.regressor_after_pooling(pooled_features_for_regression)

        # 只使用最后一层特征进行分类
        last_stage_feats = feats[-1]
        class_logits = self.classifier(last_stage_feats.view(last_stage_feats.size(0), -1))

        return class_logits, roughness_value

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the classification score and roughness value."""
        if data_samples is None:
            raise ValueError("data_samples should not be None.")

        class_logits, roughness_value = self(feats)

        # Unpack data samples safely, handling None values
        targets_cls = []
        targets_roughness = []

        for ds in data_samples:
            if ds.gt_label is not None:
                targets_cls.append(ds.gt_label)
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)

        # Ensure targets are not empty
        if not targets_cls:
            raise ValueError("No valid class labels found in data samples.")
        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")

        targets_cls = torch.cat(targets_cls)
        targets_roughness = torch.cat(targets_roughness)

        # Ensure targets_roughness is on the same device as the model's output
        device = roughness_value.device
        targets_roughness = targets_roughness.to(device)

        # Calculate classification loss
        losses = dict()
        loss_cls = self.class_loss_module(class_logits, targets_cls, avg_factor=class_logits.size(0), **kwargs)
        losses['loss_cls'] = loss_cls

        # Calculate roughness loss
        loss_roughness = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
        losses['loss_roughness'] = loss_roughness

        # Calculate MSE and MAE for roughness
        mse = F.mse_loss(roughness_value, targets_roughness)
        mae = F.l1_loss(roughness_value, targets_roughness)

        losses['mse'] = mse
        losses['mae'] = mae

        # Calculate accuracy (optional)
        if self.cal_acc:
            acc = Accuracy.calculate(class_logits, targets_cls, topk=self.topk)
            losses.update({f'accuracy_top-{k}': a for k, a in zip(self.topk, acc)})

        return losses

    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[
        DataSample]:
        """Inference without augmentation."""
        class_logits, roughness_value = self(feats)

        if data_samples is None:
            data_samples = [None for _ in range(class_logits.size(0))]

        predictions = self._get_predictions(class_logits, roughness_value, data_samples)

        # Print real labels and predictions for debugging
        for i, (prediction, data_sample) in enumerate(zip(predictions, data_samples)):
            if data_sample is not None:
                print(f"Sample {i + 1}:")
                print(f"Real Class Label: {data_sample.gt_label}")
                print(f"Predicted Class Label: {prediction.pred_label}")
                print(f"Real Roughness: {data_sample.gt_roughness}")
                print(f"Predicted Roughness: {prediction.pred_roughness}")
                print()

        return predictions

    def _get_predictions(self, class_logits: torch.Tensor, roughness_value: torch.Tensor,
                         data_samples: List[Optional[DataSample]]) -> List[DataSample]:
        """Generate prediction results for each sample."""
        predictions = []
        scores = F.softmax(class_logits, dim=1)  # Softmax for classification part

        for i, (score, roughness, data_sample) in enumerate(zip(scores, roughness_value, data_samples)):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.pred_score = score
            data_sample.pred_label = score.argmax(dim=0)
            data_sample.pred_roughness = roughness  # No softmax here for roughness

            predictions.append(data_sample)

        return predictions
