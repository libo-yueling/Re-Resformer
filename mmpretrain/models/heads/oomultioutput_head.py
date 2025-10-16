import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple, Union


@MODELS.register_module()
class OOMultiOutputHead(BaseModule):
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
        super(OOMultiOutputHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.topk = topk
        self.class_loss_module = MODELS.build(class_loss)
        self.roughness_loss_module = MODELS.build(roughness_loss)
        self.cal_acc = cal_acc

        # Initialize classifier for classification
        self.classifier = nn.Linear(in_channels, num_classes)

        # Initialize regressor for roughness regression
        self.regressor_after_pooling = nn.Linear(1920, regressor_out_channels)

        # Initialize variables for tracking losses across epochs
        self.total_class_loss = 0.0  # Accumulated classification loss
        self.total_roughness_loss = 0.0  # Accumulated roughness loss
        self.total_loss = 0.0  # Accumulated total loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: Optional[List[DataSample]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_features_for_regression = torch.cat([feat.view(feat.size(0), -1) for feat in feats], dim=1)

        # Regression for roughness
        roughness_value = self.regressor_after_pooling(pooled_features_for_regression)

        # Use only the last stage features for classification
        last_stage_feats = feats[-1]
        class_logits = self.classifier(last_stage_feats.view(last_stage_feats.size(0), -1))

        return class_logits, roughness_value

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the classification score and roughness value."""
        # Reset batch losses for the current batch
        batch_class_loss = 0.0
        batch_roughness_loss = 0.0
        tot_loss = []

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
        loss_cls = self.class_loss_module(class_logits, targets_cls, avg_factor=class_logits.size(0), **kwargs)
        batch_class_loss += loss_cls.item()  # Accumulate classification loss
        losses = {'loss_cls': loss_cls}

        # Calculate roughness loss
        loss_roughness = self.roughness_loss_module(roughness_value, targets_roughness, **kwargs)
        batch_roughness_loss += loss_roughness.item()  # Accumulate roughness loss
        losses['loss_roughness'] = loss_roughness
        tot_loss.append(loss_roughness)

        # Calculate MSE for roughness
        mse = F.mse_loss(roughness_value, targets_roughness)
        losses['mse'] = mse

        # Calculate total roughness loss
        total_roughness_loss = torch.sum(torch.stack(tot_loss))  # Sum roughness losses
        losses['total_loss_roughness'] = total_roughness_loss  # Add to total loss

        # Calculate accuracy (optional)
        if self.cal_acc:
            acc = Accuracy.calculate(class_logits, targets_cls, topk=self.topk)
            losses.update({f'accuracy_top-{k}': a for k, a in zip(self.topk, acc)})

        # Accumulate total loss across batches for the current epoch
        self.total_class_loss += batch_class_loss
        self.total_roughness_loss += batch_roughness_loss
        self.total_loss += (batch_class_loss + batch_roughness_loss)

        return losses

    def reset_epoch_loss(self):
        """Reset the accumulated losses at the beginning of each epoch."""
        self.total_class_loss = 0.0
        self.total_roughness_loss = 0.0
        self.total_loss = 0.0

    def print_epoch_loss(self, epoch: int):
        """Print the accumulated losses at the end of each epoch."""
        print(f"Epoch {epoch} - Total Classification Loss: {self.total_class_loss:.4f}")
        print(f"Epoch {epoch} - Total Roughness Loss: {self.total_roughness_loss:.4f}")
        print(f"Epoch {epoch} - Total Loss: {self.total_loss:.4f}")

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