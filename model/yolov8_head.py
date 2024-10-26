import math
from typing import List, Union, Sequence, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from model.basemodule import BaseModule
from model.networks import ConvModule
from utils import make_divisible, multi_apply

# class YOLOv8HeadModule(BaseModule):
#     """YOLOv8HeadModule head module used in `YOLOv8`.
#
#     Args:
#         num_classes (int): Number of categories excluding the background category.
#         in_channels (Union[int, Sequence[int]]): Number of channels in the input feature map.
#         widen_factor (float): Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0.
#         num_base_priors (int): The number of priors (points) at a point on the feature grid.
#         featmap_strides (Sequence[int]): Downsample factor of each feature map. Defaults to [8, 16, 32].
#         reg_max (int): Max value of integral set {0, ..., reg_max-1} in QFL setting. Defaults to 16.
#         norm_cfg (dict): Config dict for normalization layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
#         act_cfg (dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
#         init_cfg (Optional[dict]): Initialization config dict. Defaults to None.
#     """
#
#     def __init__(self,
#                  num_classes: int,
#                  in_channels: Union[int, Sequence[int]],
#                  widen_factor: float = 1.0,
#                  num_base_priors: int = 1,
#                  featmap_strides: Sequence[int] = (8, 16, 32),
#                  reg_max: int = 16,
#                  norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg: dict = dict(type='SiLU', inplace=True),
#                  init_cfg: Optional[dict] = None):
#         super().__init__()
#         self.num_classes = num_classes
#         self.featmap_strides = featmap_strides
#         self.num_levels = len(self.featmap_strides)
#         self.num_base_priors = num_base_priors
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         self.in_channels = in_channels
#         self.reg_max = reg_max
#
#         in_channels = [make_divisible(ch, widen_factor) for ch in self.in_channels]
#         self.in_channels = in_channels
#
#         self._init_layers()
#
#     def init_weights(self, prior_prob=0.01):
#         """Initialize the weight and bias of the YOLOv8 head."""
#         for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds, self.featmap_strides):
#             reg_pred[-1].bias.data[:] = 1.0  # box
#             cls_pred[-1].bias.data[:self.num_classes] = math.log(
#                 5 / self.num_classes / (640 / stride)**2)
#
#     def _init_layers(self):
#         """Initialize convolutional layers in YOLOv8 head."""
#         self.cls_preds = nn.ModuleList()
#         self.reg_preds = nn.ModuleList()
#
#         reg_out_channels = max(16, self.in_channels[0] // 4, self.reg_max * 4)
#         cls_out_channels = max(self.in_channels[0], self.num_classes)
#
#         for i in range(self.num_levels):
#             self.reg_preds.append(
#                 nn.Sequential(
#                     ConvModule(
#                         in_channels=self.in_channels[i],
#                         out_channels=reg_out_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1,
#                         norm_cfg=self.norm_cfg,
#                         act_cfg=self.act_cfg),
#                     ConvModule(
#                         in_channels=reg_out_channels,
#                         out_channels=reg_out_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1,
#                         norm_cfg=self.norm_cfg,
#                         act_cfg=self.act_cfg),
#                     nn.Conv2d(
#                         in_channels=reg_out_channels,
#                         out_channels=4 * self.reg_max,
#                         kernel_size=1)))
#             self.cls_preds.append(
#                 nn.Sequential(
#                     ConvModule(
#                         in_channels=self.in_channels[i],
#                         out_channels=cls_out_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1,
#                         norm_cfg=self.norm_cfg,
#                         act_cfg=self.act_cfg),
#                     ConvModule(
#                         in_channels=cls_out_channels,
#                         out_channels=cls_out_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1,
#                         norm_cfg=self.norm_cfg,
#                         act_cfg=self.act_cfg),
#                     nn.Conv2d(
#                         in_channels=cls_out_channels,
#                         out_channels=self.num_classes,
#                         kernel_size=1)))
#
#         proj = torch.arange(self.reg_max, dtype=torch.float)
#         self.register_buffer('proj', proj, persistent=False)
#
#     def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
#         """Forward features from the upstream network.
#
#         Args:
#             x (Tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.
#         Returns:
#             Tuple[List]: A tuple of multi-level classification scores, bbox predictions.
#         """
#         assert len(x) == self.num_levels
#         return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)
#
#     def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList) -> Tuple:
#         """Forward feature of a single scale level."""
#         b, _, h, w = x.shape
#         cls_logit = cls_pred(x)
#         bbox_dist_preds = reg_pred(x)
#         if self.reg_max > 1:
#             bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
#             bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
#             bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
#         else:
#             bbox_preds = bbox_dist_preds
#         if self.training:
#             return cls_logit, bbox_preds, bbox_dist_preds
#         else:
#             return cls_logit, bbox_preds


class YOLOv8HeadModule(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=1e-3),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg = None
                 ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layer()

    def _init_layer(self):
        self.cls_preds = nn.ModuleList()
        self.

class YOLOv8Head(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self._raw_positive_infos = dict()
        self.head_module = YOLOv8Head(train_cfg=train_cfg, test_cfg=test_cfg)