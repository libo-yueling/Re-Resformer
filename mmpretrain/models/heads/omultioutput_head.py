import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple, Union
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

@MODELS.register_module()
class OMultiOutputHead(BaseModule):
    """Multi-Output head for classification and regression (with linear regression for roughness)."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 class_loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 roughness_loss: dict = dict(type='PTLoss', loss_weight=1.0, reduction='sum'),
                 regressor_out_channels: int = 1,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(OMultiOutputHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.topk = topk
        self.class_loss_module = MODELS.build(class_loss)
        self.roughness_loss_module = MODELS.build(roughness_loss)
        self.cal_acc = cal_acc

        # Initialize classifier for classification
        self.classifier = nn.Linear(in_channels, num_classes)

        # Initialize regressor for roughness regression
        self.regressor_after_pooling = nn.Linear(1920, regressor_out_channels)

        # Initialize GBT model
        self.gbt_model = GradientBoostingRegressor()

        # Placeholder for feature selection
        self.selected_features_indices = None

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将所有层的特征进行拼接
        pooled_features_for_regression = torch.cat([feat.view(feat.size(0), -1) for feat in feats], dim=1)

        # 动态阈值选择和特征选择
        if self.selected_features_indices is None:
            self.dynamic_threshold_selection(pooled_features_for_regression, data_samples)

        selected_features = pooled_features_for_regression[:, self.selected_features_indices]

        # 回归粗糙度
        roughness_value = self.regressor_after_pooling(selected_features)

        # 只使用最后一层特征进行分类
        last_stage_feats = feats[-1]
        class_logits = self.classifier(last_stage_feats.view(last_stage_feats.size(0), -1))

        return class_logits, roughness_value

    def dynamic_threshold_selection(self, X, data_samples: List[DataSample]):
        """Dynamically select the best threshold based on validation performance."""
        if data_samples is None or len(data_samples) == 0:
            return

        targets_roughness = []
        for ds in data_samples:
            if ds.gt_roughness is not None:
                targets_roughness.append(ds.gt_roughness)

        if not targets_roughness:
            raise ValueError("No valid roughness values found in data samples.")

        targets_roughness = torch.cat(targets_roughness).cpu().numpy()
        X = X.cpu().numpy()

        # Assuming a split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, targets_roughness, test_size=0.2, random_state=42)

        self.gbt_model.fit(X_train, y_train)
        importances = self.gbt_model.feature_importances_
        thresholds = np.linspace(0, np.max(importances), 100)

        best_threshold = 0
        min_error = float('inf')

        for threshold in thresholds:
            selected_indices = np.where(importances > threshold)[0]
            if len(selected_indices) == 0:
                continue

            X_train_selected = X_train[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]

            self.gbt_model.fit(X_train_selected, y_train)
            y_pred = self.gbt_model.predict(X_val_selected)
            error = mean_squared_error(y_val, y_pred)

            if error < min_error:
                min_error = error
                best_threshold = threshold

        self.threshold = best_threshold
        self.selected_features_indices = np.where(importances > self.threshold)[0]

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

        # 计算总的粗糙度损失 (如果你有多个样本的情况)
        total_roughness_loss = torch.sum(loss_roughness)  # 对所有的粗糙度损失进行求和
        losses['total_loss_roughness'] = total_roughness_loss  # 将总的粗糙度损失加入到最终损失中

        # Calculate MSE for roughness
        mse = F.mse_loss(roughness_value, targets_roughness)
        losses['mse'] = mse

        # Calculate accuracy (optional)
        if self.cal_acc:
            acc = Accuracy.calculate(class_logits, targets_cls, topk=self.topk)
            losses.update({f'accuracy_top-{k}': a for k, a in zip(self.topk, acc)})

        return losses

    def predict(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[Optional[DataSample]]] = None) -> List[DataSample]:
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
            data_sample.pred_roughness = roughness

            predictions.append(data_sample)

        return predictions
