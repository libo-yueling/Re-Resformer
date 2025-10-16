import os
import datetime
import pandas as pd
import torch
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample


@HOOKS.register_module()
class ExcelLoggerHook(Hook):
    """Excel Logger Hook for saving prediction results to an Excel file.

    This hook collects the roughness ground truth and predicted values from DataSample objects
    during validation, computes the MAPE for each sample, and writes the results to an Excel file
    after validation.

    Args:
        out_dir (str, optional): Directory where the Excel file will be saved.
            If None, a folder will be created under 'E:/classiyf-module/mmpretrain-main/tools'
            with the current timestamp as its name.
        filename (str): Name of the Excel file. Defaults to "roughness_predictions.xlsx".
    """

    def __init__(self,
                 out_dir: Optional[str] = None,
                 filename: str = "roughness_predictions.xlsx"):
        self.out_dir = out_dir
        self.filename = filename
        self.results = []
        self.log_folder = None
        self.filepath = None  # 用于存储最终的 Excel 文件路径

    def before_val(self, runner: Runner) -> None:
        """Called before validation starts. Create the log folder and initialize the file path."""
        if self.out_dir is None:
            base_dir = "E:/classiyf-module/mmpretrain-main/tools"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_folder = os.path.join(base_dir, timestamp)
            # 用第一次生成的时间戳来命名 Excel 文件，后续都使用同一个文件
            filename_with_timestamp = f"roughness_predictions_{timestamp}.xlsx"
        else:
            self.log_folder = self.out_dir
            filename_with_timestamp = self.filename

        os.makedirs(self.log_folder, exist_ok=True)
        self.filepath = os.path.join(self.log_folder, filename_with_timestamp)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DataSample]) -> None:
        """Collect results from each validation batch."""
        for idx, ds in enumerate(outputs):
            # 获取真实值和预测值
            if ds.gt_roughness is not None:
                gt_val = ds.gt_roughness.item() if isinstance(ds.gt_roughness, torch.Tensor) else ds.gt_roughness
            else:
                gt_val = None
            if ds.pred_roughness is not None:
                pred_val = ds.pred_roughness.item() if isinstance(ds.pred_roughness, torch.Tensor) else ds.pred_roughness
            else:
                pred_val = None

            # 计算 MAPE（当真实值不为0时）
            if gt_val is not None and pred_val is not None and gt_val != 0:
                mape = abs(gt_val - pred_val) / abs(gt_val) * 100
            else:
                mape = None

            self.results.append({
                "Epoch": runner.epoch,  # 使用 runner.epoch 获取当前的 epoch
                "Batch": batch_idx,
                "Image Index": idx,
                "True Roughness": gt_val,
                "Predicted Roughness": pred_val,
                "MAPE (%)": mape
            })

    def after_val(self, runner: Runner) -> None:
        """After validation, write the collected results to the Excel file."""
        if not self.results:
            print("No results collected; skipping Excel export.")
            return

        df_new = pd.DataFrame(self.results)

        # 如果 Excel 文件已存在，则加载已有数据并追加新数据，否则直接使用新数据
        if os.path.exists(self.filepath):
            df_old = pd.read_excel(self.filepath)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_excel(self.filepath, index=False)
        print(f"Excel file saved to {self.filepath}")

        # 清空 results 以便下次验证重新收集数据
        self.results = []
