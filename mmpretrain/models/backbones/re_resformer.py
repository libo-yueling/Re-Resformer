# Copyright (c) OpenMMLab. All rights reserved.
'''
用ResfoemerEncoder替代resnet的stage2/4

'''
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone

eps = 1.0e-5


# 1. 定义注意力模块，threshold 和 K 均为可学习参数
class DADMAttention(nn.Module):
    def __init__(self, channels, reduction=16, init_threshold=1.2, init_k=10.0):
        """
        :param channels: 输入特征通道数
        :param reduction: 用于全连接层的降维率
        :param init_threshold: 阈值初始值（当局部特征值超过均值的比例）
        :param init_k: Sigmoid 中控制陡峭程度的初始值
        threshold阈值：局部值与均值之比。init_threshold=1.2,意味着当局部特征值超过其均值 1.2 倍时，就可能认定为异常区域
        """
        super(DADMAttention, self).__init__()
        self.channels = channels
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        # 将 threshold 和 K 设为可学习参数
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float))
        self.k = nn.Parameter(torch.tensor(init_k, dtype=torch.float))

    def forward(self, x):
        """
        :param x: 输入特征图 [B, C, H, W]
        :return: 加权后的特征图及注意力掩码（用于可视化）
        """
        B, C, H, W = x.size()

        # 全局平均池化得到全局均值向量
        global_mean = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        attn_global = F.relu(self.fc1(global_mean))
        attn_global = torch.sigmoid(self.fc2(attn_global)).view(B, C, 1, 1)

        # 计算局部均值（这里简单使用空间平均）
        eps = 1e-6
        local_mean = x.mean(dim=(2, 3), keepdim=True) + eps
        ratio = x / local_mean  # 得到局部与均值的比值

        # 使用软阈值：输出范围 [0, 1]，接近 1 表示正常区域，接近 0 表示缺陷区域
        defect_mask = torch.sigmoid(self.k * (self.threshold - ratio))

        # 融合全局信息与缺陷抑制
        attn = attn_global * defect_mask

        # 返回加权后的特征图和用于可视化的掩码
        out = x * attn
        return out


# 2. DSConv 模块（深度可分离卷积
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSConv, self).__init__()
        # 逐通道卷积：groups 设置为 in_channels 实现每个通道独立卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        # 逐点卷积：1x1 卷积，用于通道整合和调整输出通道数
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gelu(x)
        return x


# 3. KAN 模块（包含一个卷积层和 sin 激活）
class KAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(KAN, self).__init__()
        # KAN 的非线性变换部分
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = torch.sin(x)
        x = self.gelu(x)
        return x


# 4. Resfoemer Encoder 的定义
# 五个部分：DSConv1 -> LN1 -> DSConv2 -> LN2 -> KAN
class ResformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ResformerEncoder, self).__init__()
        # 第一部分
        self.dsconv1 = DSConv(in_channels, hidden_channels)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=hidden_channels)
        self.ln = nn.LayerNorm(hidden_channels)
        self.attten1 = DADMAttention(hidden_channels)
        # 如果 in_channels 与 hidden_channels 不同，可以添加投影
        if in_channels != hidden_channels:
            self.proj1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        else:
            self.proj1 = nn.Identity()

        self.dsconv2 = DSConv(hidden_channels, hidden_channels)

        # 第一残差分支投影（如果需要的话，这里通常是相同通道数就可以直接加）
        # 第二部分
        # 为了匹配从 hidden_channels 到 out_channels 的转换
        self.proj2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1)
        self.kan = KAN(hidden_channels, out_channels)
        # 此 conv2 可以用作另一条路径（根据需要）
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        # 如果需要先对输入进行投影
        x = self.proj1(x)
        x = self.attten1(x)
        x = self.dsconv1(x)
        residual = x.clone()  # 残差分支1
        x = self.gn(x)
        x = self.dsconv2(x)
        # 第一残差连接
        x += residual

        residual = x.clone()  # 残差分支2
        x = self.gn(x)
        x = self.kan(x)
        # 对第二个残差分支进行通道投影，使其与 x 匹配
        x += self.proj2(residual)
        return x


class BasicBlock(BaseModule):
    """BasicBlock for ReResformer.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ReResformer.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ReResformer style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        drop_path_rate (float or list): stochastic depth rate.
            Default: 0.
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks

        assert len(drop_path_rate
                   ) == num_blocks, 'Please check the length of drop_path_rate'

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=drop_path_rate[0],
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate[i],
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@MODELS.register_module()
class ReResformer(BaseBackbone):
    """ReResformer backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ReResformer
        >>> import torch
        >>> self = ReResformer(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[dict(type='Kaiming', layer=['Conv2d']),
                           dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])],
                 drop_path_rate=0.0):
        super(ReResformer, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for reresformer')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        total_depth = sum(stage_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        if depth in [18, 34]:
            # self.layer1 = ResformerEncoder(in_channels=stem_channels, hidden_channels=stem_channels, out_channels=64)
            self.layer2 = ResformerEncoder(in_channels=64, hidden_channels=64, out_channels=128)
            # self.layer3 = ResformerEncoder(in_channels=128, hidden_channels=128, out_channels=256)
            self.layer4 = ResformerEncoder(in_channels=256, hidden_channels=256, out_channels=512)
        elif depth in [50, 101, 152]:
            # self.layer1 = ResformerEncoder(in_channels=stem_channels, hidden_channels=stem_channels, out_channels=256)
            self.layer2 = ResformerEncoder(in_channels=256, hidden_channels=256, out_channels=512)
            # self.layer3 = ResformerEncoder(in_channels=512, hidden_channels=512, out_channels=1024)
            self.layer4 = ResformerEncoder(in_channels=1024, hidden_channels=1024, out_channels=2048)

        self._freeze_stages()
        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ReResformer, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
                    
    def forward(self, x, cond=None):
        """修改 forward 接口，增加 cond 参数用于传递 RSN 条件信息"""
        layer_features = []

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.maxpool(x)

        for layer_name in self.res_layers:
            layer = getattr(self, layer_name)
            x = layer(x)
            layer_features.append(x)
        return tuple(layer_features)


    def train(self, mode=True):
        super(ReResformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ReResformer.

        ReResformer stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1


@MODELS.register_module()
class ReResformerV1c(ReResformer):
    """ReResformerV1c backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ReResformer(ReResformerV1b), ReResformerV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.
    """

    def __init__(self, **kwargs):
        super(ReResformerV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@MODELS.register_module()
class ReResformerV1d(ReResformer):
    """ReResformerV1d backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ReResformer(ReResformerV1b), ReResformerV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ReResformerV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
