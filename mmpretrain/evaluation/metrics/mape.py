from typing import List, Optional, Sequence, Union
import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS

import pandas as pd
import datetime
import os


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class MAPE(BaseMetric):
    r"""Mean Absolute Percentage Error (MAPE) evaluation metric for regression tasks.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or 'gpu'.
            Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmpretrain.evaluation import MAPE
        >>> # -------------------- Basic Usage --------------------
        >>> y_pred = torch.tensor([1.2, 2.3, 3.4, 4.5])
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> MAPE.calculate(y_pred, y_true)
        tensor(12.5)  # 示例输出
        >>>
        >>> # ------------------- Use with Evaluator -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(1.0).set_pred_score(torch.tensor(1.2)),
        ...     DataSample().set_gt_label(2.0).set_pred_score(torch.tensor(2.3))
        ... ]
        >>> evaluator = Evaluator(metrics=MAPE())
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(2)
    """
    default_prefix: Optional[str] = 'mape'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.save_dir = 'E:/classiyf-module/mmpretrain-main/tools'
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = None

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            # 收集预测值和真实值
            result['pred_roughness'] = data_sample['pred_roughness'].cpu()
            result['gt_roughness'] = data_sample['gt_roughness'].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        # Concatenate predictions and ground truths
        pred = torch.cat([res['pred_roughness'] for res in results])
        target = torch.cat([res['gt_roughness'] for res in results])

        # Validate dimensions match
        assert pred.shape == target.shape, \
            f"Prediction shape {pred.shape} doesn't match target {target.shape}"

        # Calculate MAPE
        mape = self.calculate(pred, target)

        # Save MAPE result to Excel file (if needed)
        if self.filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.filename = os.path.join(self.save_dir, f'mape_{timestamp}.xlsx')

        new_data = pd.DataFrame({'MAPE': [mape.item()]})
        if os.path.exists(self.filename):
            existing_data = pd.read_excel(self.filename)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.to_excel(self.filename, index=False, engine='openpyxl')
        else:
            new_data.to_excel(self.filename, index=False, engine='openpyxl')

        print(f"MAPE results saved to {self.filename}")

        return {'mape': mape.item()}

    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray, Sequence],
                  target: Union[torch.Tensor, np.ndarray, Sequence]) -> torch.Tensor:
        """Calculate Mean Absolute Percentage Error (MAPE).

        Args:
            pred (Union[torch.Tensor, np.ndarray, Sequence]): Predictions.
            target (Union[torch.Tensor, np.ndarray, Sequence]): Ground truths.

        Returns:
            torch.Tensor: The computed MAPE value.
        """
        # 转换为张量
        pred = to_tensor(pred).float()
        target = to_tensor(target).float()

        # 验证维度一致性
        assert pred.shape == target.shape, \
            f"Prediction shape {pred.shape} doesn't match target {target.shape}"

        # 计算平均绝对百分比误差
        return torch.mean(torch.abs((target - pred) / target)) * 100