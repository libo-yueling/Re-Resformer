
from typing import List, Optional, Sequence, Union
import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


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
class MSE(BaseMetric):
    r"""Mean Squared Error (MSE) evaluation metric for regression tasks.

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
        >>> from mmpretrain.evaluation import MSE
        >>> # -------------------- Basic Usage --------------------
        >>> y_pred = torch.tensor([1.2, 2.3, 3.4, 4.5])
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> MSE.calculate(y_pred, y_true)
        tensor(0.0850)
        >>>
        >>> # ------------------- Use with Evaluator -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(1.0).set_pred_score(torch.tensor(1.2)),
        ...     DataSample().set_gt_label(2.0).set_pred_score(torch.tensor(2.3)))
        ... ]
        >>> evaluator = Evaluator(metrics=MSE())
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(2)
        {'mse': 0.06500000000000002}
    """
    default_prefix: Optional[str] = 'mse'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            result['pred_roughness'] = data_sample['pred_roughness'].cpu()
            result['gt_roughness'] = data_sample['gt_roughness'].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            dict: The computed metrics with keys like 'mse'.
        """
        pred = torch.cat([res['pred_roughness'] for res in results])
        target = torch.cat([res['gt_roughness'] for res in results])

        assert pred.shape == target.shape, \
            f"Prediction shape {pred.shape} doesn't match target {target.shape}"

        mse = self.calculate(pred, target)
        return {'mse': mse.item()}

    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray, Sequence],
                  target: Union[torch.Tensor, np.ndarray, Sequence]) -> torch.Tensor:
        """Calculate Mean Squared Error (MSE).

        Args:
            pred (Union[torch.Tensor, np.ndarray, Sequence]): Predictions.
            target (Union[torch.Tensor, np.ndarray, Sequence]): Ground truths.

        Returns:
            torch.Tensor: The computed MSE value.
        """
        pred = to_tensor(pred).float()
        target = to_tensor(target).float()

        assert pred.shape == target.shape, \
            f"Prediction shape {pred.shape} doesn't match target {target.shape}"

        return torch.mean((pred - target) ** 2)
