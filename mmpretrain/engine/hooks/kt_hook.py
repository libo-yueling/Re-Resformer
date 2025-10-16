# kt_hook.py

import os
import datetime
import pandas as pd
import torch
from typing import Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmpretrain.registry import HOOKS

# 根据你的项目结构修改导入路径：
# 假设 LearnableGlobalAttention 定义在 mmpretrain/models/backbones/resformer.py
from mmpretrain.models.backbones.resformer import DADMAttention


@HOOKS.register_module()
class KTExcelHook(Hook):
    """KTExcelHook: 在每个训练 epoch 结束时，将所有 LearnableGlobalAttention 模块的
    k 和 threshold 参数写入同一个 Excel 文件。

    每个 epoch 会往同一个 Excel 里追加一行，列包括：
        - Epoch
        - ModuleName
        - Threshold
        - K

    Args:
        out_dir (str, optional): 保存 Excel 文件的目录。如果为 None，则在
            "E:/classiyf-module/mmpretrain-main/tools/YYYYMMDD_HHMMSS/" 下新建文件夹。
        filename (str): Excel 文件名，默认为 "lga_params.xlsx"。
    """

    def __init__(self,
                 out_dir: Optional[str] = None,
                 filename: str = "lga_params.xlsx"):
        super().__init__()
        self.out_dir = out_dir
        self.filename = filename
        self.log_folder = None   # 存放生成的目录路径
        self.filepath = None     # 完整的 Excel 文件路径

    def before_train(self, runner: Runner) -> None:
        """训练开始前，创建保存目录并确定 Excel 路径。"""
        if self.out_dir is None:
            base_dir = "E:/classiyf-module/mmpretrain-main/tools"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_folder = os.path.join(base_dir, timestamp)
        else:
            self.log_folder = self.out_dir

        os.makedirs(self.log_folder, exist_ok=True)
        self.filepath = os.path.join(self.log_folder, self.filename)

        # 如果文件不存在，先写入表头
        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=["Epoch", "ModuleName", "Threshold", "K"])
            df.to_excel(self.filepath, index=False)

    def after_train_epoch(self, runner: Runner) -> None:
        """在每个 epoch 结束后，收集所有 LGA 实例的 k 和 threshold，并写入 Excel。"""
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

        # 如果文件已存在，读取并追加；否则直接写
        if os.path.exists(self.filepath):
            df_old = pd.read_excel(self.filepath)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_excel(self.filepath, index=False)
        runner.logger.info(f"KTExcelHook saved LGA params for epoch {epoch_idx} to {self.filepath}")
