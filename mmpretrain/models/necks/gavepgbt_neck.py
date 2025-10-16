# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mmpretrain.registry import MODELS

class CustomGBT(GradientBoostingRegressor):
    def __init__(self,
                 loss='squared_error',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha)

    def fit(self, X, y, **fit_params):
        # 在此处添加自定义的 fit 方法逻辑
        return super().fit(X, y, **fit_params)

    def predict(self, X):
        # 在此处添加自定义的 predict 方法逻辑
        return super().predict(X)


@MODELS.register_module()
class GAVEPGBT(BaseModule):
    """Neck with Global Average Pooling and Custom Gradient Boosting Tree for feature selection.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        gap_dim (int): Dimensions of each sample channel, can be one of
            {0, 1, 2, 3}. Defaults to 0.
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Defaults to dict(type='BN1d').
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Defaults to None.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
        threshold (float, optional): Importance score threshold for feature selection.
            Defaults to 0.01.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 gap_dim: int = 0,
                 norm_cfg: Optional[dict] = dict(type='BN1d'),
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 threshold: float = 0.01):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)
        self.threshold = threshold
        self.gbt_model = CustomGBT()
        self.selected_features_indices = None

        assert gap_dim in [0, 1, 2, 3], 'GlobalAveragePooling dim only ' \
            f'support {0, 1, 2, 3}, get {gap_dim} instead.'
        if gap_dim == 0:
            self.gap = nn.Identity()
        elif gap_dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif gap_dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.norm = None
        self.act = None
        self.fc = None

    def fit_gbt(self, X, y):
        """Fit the GBT model to the features and target."""
        self.gbt_model.fit(X, y)
        importances = self.gbt_model.feature_importances_
        self.selected_features_indices = np.where(importances > self.threshold)[0]

        # 动态创建全连接层
        self.fc = nn.Linear(in_features=len(self.selected_features_indices), out_features=self.out_channels)

        if self.norm_cfg:
            self.norm = build_norm_layer(self.norm_cfg, self.out_channels)[1]
        else:
            self.norm = nn.Identity()

        if self.act_cfg:
            self.act = build_activation_layer(self.act_cfg)
        else:
            self.act = nn.Identity()

    def dynamic_threshold_selection(self, X, y, X_val, y_val):
        """Dynamically select the best threshold based on validation performance."""
        importances = self.gbt_model.feature_importances_
        thresholds = np.linspace(0, np.max(importances), 100)

        best_threshold = 0
        min_error = float('inf')

        for threshold in thresholds:
            selected_indices = np.where(importances > threshold)[0]
            if len(selected_indices) == 0:
                continue

            X_train_selected = X[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]

            self.gbt_model.fit(X_train_selected, y)
            y_pred = self.gbt_model.predict(X_val_selected)
            error = mean_squared_error(y_val, y_pred)

            if error < min_error:
                min_error = error
                best_threshold = threshold

        self.threshold = best_threshold
        self.selected_features_indices = np.where(importances > self.threshold)[0]

        # 动态创建全连接层
        self.fc = nn.Linear(in_features=len(self.selected_features_indices), out_features=self.out_channels)

    def forward(self, inputs: Union[Tuple, torch.Tensor]) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Union[Tuple, torch.Tensor]): The features extracted from
                the backbone. Multiple stage inputs are acceptable but only
                the last stage will be used.

        Returns:
            Tuple[torch.Tensor]: A tuple of output features.
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `GAVEPGBT` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        x = self.gap(inputs)
        x = x.view(x.size(0), -1)

        # 如果没有选择特征索引，则返回池化后的特征
        if self.selected_features_indices is None:
            return (x, )

        # 选择重要特征
        selected_features = x[:, self.selected_features_indices]
        out = self.act(self.norm(self.fc(selected_features)))
        return (out, )

if __name__ == '__main__':
    # 假设我们有输入特征 x 和目标变量 y
    x = torch.randn(10, 512, 7, 7)  # 示例输入特征
    y = np.random.randn(10)  # 示例目标变量

    # 初始化并训练 GAVEPGBT 模型
    gmaxpgbt = GAVEPGBT(in_channels=512, out_channels=256, gap_dim=2)
    pooled_features = gmaxpgbt.gap(x).view(x.size(0), -1).detach().numpy()
    X_train, X_val, y_train, y_val = train_test_split(pooled_features, y, test_size=0.2, random_state=42)
    gmaxpgbt.fit_gbt(X_train, y_train)
    gmaxpgbt.dynamic_threshold_selection(X_train, y_train, X_val, y_val)

    # 前向传播，选择重要特征
    selected_features = gmaxpgbt(x)
    print(selected_features[0].shape)

