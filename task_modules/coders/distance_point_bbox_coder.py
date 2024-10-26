from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from typing import Optional, Union, Sequence

class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder.

    Args:
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to False.
    """

    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""



class DistancePointBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether to clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border: Optional[bool] = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_border = clip_border

    def encode(self,
               points: Tensor,
               gt_bboxes: Tensor,
               max_dis: Optional[float] = None,
               eps: float = 0.1) -> Tensor:
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 4), The format is "xyxy".
            max_dis (float, optional): Upper bound of the distance. Default is None.
            eps (float): A small value to ensure target < max_dis, instead <=.
                Default is 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0), "Points and gt_bboxes must have the same number of elements."
        assert points.size(-1) == 2, "Points should have shape (N, 2)."
        assert gt_bboxes.size(-1) == 4, "gt_bboxes should have shape (N, 4)."

        # 计算点到边界框的距离
        left = points[:, 0] - gt_bboxes[:, 0]  # x - x1
        top = points[:, 1] - gt_bboxes[:, 1]   # y - y1
        right = gt_bboxes[:, 2] - points[:, 0]  # x2 - x
        bottom = gt_bboxes[:, 3] - points[:, 1]  # y2 - y

        distances = torch.stack([top, bottom, left, right], dim=-1)

        # 如果提供了最大距离，进行裁剪
        if max_dis is not None:
            distances = torch.clamp(distances - eps, min=0, max=max_dis)

        return distances

    def decode(self,
               points: Tensor,
               pred_bboxes: Tensor,
               max_shape: Optional[Union[Sequence[int], Tensor, Sequence[Sequence[int]]]] = None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (top, bottom, left, right). Shape (N, 4).
            max_shape (Sequence[int] or Tensor or Sequence[Sequence[int]], optional):
                Maximum bounds for boxes, specifies (H, W). Default is None.

        Returns:
            Union[Tensor, BaseBoxes]: Boxes with shape (N, 4).
        """
        assert points.size(0) == pred_bboxes.size(0), "Points and pred_bboxes must have the same number of elements."
        assert points.size(-1) == 2, "Points should have shape (N, 2)."
        assert pred_bboxes.size(-1) == 4, "pred_bboxes should have shape (N, 4)."

        # 解码距离到边界框坐标
        top = points[:, 1] + pred_bboxes[:, 0]
        bottom = points[:, 1] + pred_bboxes[:, 1]
        left = points[:, 0] + pred_bboxes[:, 2]
        right = points[:, 0] + pred_bboxes[:, 3]

        bboxes = torch.stack([left, top, right, bottom], dim=-1)

        # 如果需要裁剪边界框
        if self.clip_border and max_shape is not None:
            height, width = max_shape[:2] if isinstance(max_shape, Sequence) else max_shape
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(min=0, max=width)  # left, right
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(min=0, max=height)  # top, bottom

        return bboxes