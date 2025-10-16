import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures.data_sample import DataSample
from typing import List, Optional, Tuple, Union

@MODELS.register_module()
class WMultiOutputHead(BaseModule):
    """Multi-Output head for classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes for classification.
        class_loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=0.5)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, 5)``.
        cal_acc (bool): Whether to calculate accuracy during training.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 class_loss: dict = dict(type='CrossEntropyLoss', loss_weight=0.5),
                 topk: Union[int, Tuple[int]] = (1, 5),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(WMultiOutputHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.topk = topk
        self.class_loss_module = MODELS.build(class_loss)
        self.cal_acc = cal_acc

        # Initialize classifier
        self.classifier = nn.Linear(in_channels, num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head."""
        return feats[-1]

    def forward(self, feats: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> torch.Tensor:
        """The forward process for classification."""
        last_stage_feats = self.pre_logits(feats)
        class_logits = self.classifier(last_stage_feats)
        return class_logits

    def loss(self, feats: torch.Tensor, data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from the classification score."""
        if data_samples is None:
            raise ValueError("data_samples should not be None.")

        class_logits = self(feats)

        # Unpack data samples safely, handling None values
        targets_cls = []
        for ds in data_samples:
            if ds.gt_label is not None:
                targets_cls.append(ds.gt_label)

        # Ensure targets are not empty
        if not targets_cls:
            raise ValueError("No valid class labels found in data samples.")

        targets_cls = torch.cat(targets_cls)

        # Calculate classification loss
        losses = dict()
        loss_cls = self.class_loss_module(class_logits, targets_cls, avg_factor=class_logits.size(0), **kwargs)
        losses['loss_cls'] = loss_cls

        # Calculate accuracy (optional)
        if self.cal_acc:
            acc = Accuracy.calculate(class_logits, targets_cls, topk=self.topk)
            losses.update({f'accuracy_top-{k}': a for k, a in zip(self.topk, acc)})

        return losses

    def predict(self, feats: torch.Tensor, data_samples: Optional[List[Optional[DataSample]]] = None) -> List[DataSample]:
        """Inference without augmentation."""
        class_logits = self(feats)

        if data_samples is None:
            data_samples = [None for _ in range(class_logits.size(0))]

        predictions = self._get_predictions(class_logits, data_samples)
        return predictions

    def _get_predictions(self, class_logits: torch.Tensor, data_samples: List[Optional[DataSample]]) -> List[DataSample]:
        """Generate prediction results for each sample."""
        predictions = []
        scores = F.softmax(class_logits, dim=1)  # Softmax for classification part

        for i, (score, data_sample) in enumerate(zip(scores, data_samples)):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.pred_score = score
            data_sample.pred_label = score.argmax(dim=0)
            predictions.append(data_sample)

        return predictions
