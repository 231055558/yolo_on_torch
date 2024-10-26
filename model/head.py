from typing import Optional, List

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from model.basemodule import BaseModule


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg):
        super().__init__(init_cfg=init_cfg)
        self._raw_positive_infos = dict()

    def init_weights(self):
        super().init_weights()
        for m in self.modules():
            if hasattr(m, 'conv_offset'):
                nn.init.constant_(m.conv_offset, 0)

    def forward(self, x):
        """Forward function for the dense head. Needs to be defined in subclasses."""
        raise NotImplementedError

class YOLOv5Head(BaseDenseHead):
    def __init__(self,
                 head_module: dict,
                 prior_generator: dict = dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(10, 13), (16, 30), (33, 23)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(116, 90), (156, 198), (373, 326)]],
                     strides=[8, 16, 32]),
                 bbox_coder: dict = dict(type='YOLOv5BBoxCoder'),
                 loss_cls: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=0.5),
                 loss_bbox: dict = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xywh',
                     eps=1e-7,
                     reduction='mean',
                     loss_weight=0.05,
                     return_iou=True),
                 loss_obj: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.head_module =


