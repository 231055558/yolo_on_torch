import copy
import math
from typing import Union, Sequence, Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from model.structures import InstanceData
from task_modules.coders.distance_point_bbox_coder import DistancePointBBoxCoder
from task_modules.prior_generators.point_generator import MlvlPointGenerator
from loss.cross_entropy_loss import CrossEntropyLoss
from loss.iou_loss import IoULoss
from loss.gfocal_loss import DistributionFocalLoss
from model.basemodule import BaseModule
from model.networks import ConvModule
from utils import make_divisible, multi_apply, gt_instances_preprocess, get_dist_info, filter_scores_and_topk
from task_modules.assigners.batch_task_aligned_assigner import BatchTaskAlignedAssigner

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

        self._init_layers()

    def _init_layers(self):
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4)
        )
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)))

            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1)))
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def init_weights(self, prior_prob=0.01):
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)

    def forward(self, x):
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)

    def forward_single(self, x, cls_pred,
                       reg_pred):
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


class YOLOv8Head(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self._raw_positive_infos = dict()
        self.head_module = YOLOv8HeadModule(num_classes=num_classes,
                                            in_channels=in_channels,
                                            widen_factor=widen_factor,
                                            featmap_strides=featmap_strides,
                                            reg_max=reg_max)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_cls: nn.Module = CrossEntropyLoss(use_sigmoid=True, reduction='none', loss_weight=0.5)
        self.loss_bbox: nn.Module = IoULoss(iou_mode='ciou', bbox_format='xyxy', reduction='sum', loss_weight=0.75, return_iou=False)
        self.loss_obj = None

        self.prior_generator = MlvlPointGenerator(offset=0.5, strides=self.featmap_strides)
        self.bbox_coder = DistancePointBBoxCoder()
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.ignore_iof_thr = ignore_iof_thr
        self.obj_level_weights = obj_level_weights

        self.special_init()

        self.loss_dfl = DistributionFocalLoss(reduction='mean', loss_weight=0.375)

    def special_init(self):
        # assert len(self.obj_level_weights) == len(
        #     self.featmap_strides) == self.num_levels
        # priors_base_sizes = torch.tensor(
        #     self.prior_generator.base_sizes, dtype=torch.float)
        # featmap_strides = torch.tensor(
        #     self.featmap_strides, dtype=torch.float)[:, None, None]
        # self.register_buffer(
        #     'priors_base_sizes',
        #     priors_base_sizes / featmap_strides,
        #     persistent=False)
        #
        # grid_offset = torch.tensor([
        #     [0, 0],  # center
        #     [1, 0],  # left
        #     [0, 1],  # up
        #     [-1, 0],  # right
        #     [0, -1],  # bottom
        # ]).float()
        # self.register_buffer(
        #     'grid_offset', grid_offset[:, None], persistent=False)
        #
        # prior_inds = torch.arange(self.num_base_priors).float().view(
        #     self.num_base_priors, 1)
        # self.register_buffer('prior_inds', prior_inds, persistent=False)
        if self.train_cfg:
            self.assigner = BatchTaskAlignedAssigner(num_classes=self.num_classes, use_ciou=True, topk=10, alpha=0.5, beta=6.0, eps=1e-09)
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def init_weights(self):
        super().init_weights()

    def forward(self, x):
        return self.head_module(x)

    def loss(self, x, batch_data_samples):
        outs = self(x)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances,
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore = None) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)

    def predict(self,
                x,
                batch_size_sample,
                rescale=False):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_size_sample
        ]
        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional = None,
                        rescale: bool = True,
                        with_nms: bool = True):
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list