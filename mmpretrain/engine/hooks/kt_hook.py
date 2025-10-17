# kt_hook.py

import os
import datetime
import pandas as pd
import torch
from typing import Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmpretrain.registry import HOOKS

# Modify the import path according to your project structure:
from mmpretrain.models.backbones.resformer import DADMAttention


@HOOKS.register_module()
class KTExcelHook(Hook):
    """KTExcelHook: At the end of each training epoch, write the k and threshold parameters of all LearnableGlobalAttention modules into the same Excel file.

    Each epoch appends a row to the same Excel file, with columns including：
        - Epoch
        - ModuleName
        - Threshold
        - K
    """

    def __init__(self,
                 out_dir: Optional[str] = None,
                 filename: str = "lga_params.xlsx"):
        super().__init__()
        self.out_dir = out_dir
        self.filename = filename
        self.log_folder = None   
        self.filepath = None     

    def before_train(self, runner: Runner) -> None:
        if self.out_dir is None:
            # Modify according to the tools path of your own project
            base_dir = "E:/classiyf-module/mmpretrain-main/tools"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_folder = os.path.join(base_dir, timestamp)
        else:
            self.log_folder = self.out_dir

        os.makedirs(self.log_folder, exist_ok=True)
        self.filepath = os.path.join(self.log_folder, self.filename)

        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=["Epoch", "ModuleName", "Threshold", "K"])
            df.to_excel(self.filepath, index=False)

    def after_train_epoch(self, runner: Runner) -> None:
                model = runner.model.module if hasattr(runner.model, "module") else runner.model
        epoch_idx = runner.epoch

        rows = []
        for name, module in model.named_modules():
            if isinstance(module, LearnableGlobalAttention):
                t_val = module.threshold.item() if isinstance(module.threshold, torch.Tensor) else float(module.threshold)
                k_val = module.k.item() if isinstance(module.k, torch.Tensor) else float(module.k)
                rows.append({
                    "Epoch": epoch_idx,
                    "ModuleName": name,
                    "Threshold": t_val,
                    "K": k_val
                })

        if not rows:
            return

        df_new = pd.DataFrame(rows)

        if os.path.exists(self.filepath):
            df_old = pd.read_excel(self.filepath)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_excel(self.filepath, index=False)
        runner.logger.info(f"KTExcelHook saved LGA params for epoch {epoch_idx} to {self.filepath}")
