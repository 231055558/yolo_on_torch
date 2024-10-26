import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """
        CrossEntropyLoss.

        Args:
            use_sigmoid (bool): Whether the prediction uses sigmoid.
            use_mask (bool): Whether to use mask cross entropy loss.
            reduction (str): Options are "none", "mean" and "sum".
            class_weight (list[float]): Weight of each class.
            ignore_index (int): The label index to be ignored.
            loss_weight (float): Weight of the loss.
            avg_non_ignore (bool): Whether to average loss over non-ignored targets.
        """
        super(CrossEntropyLoss, self).__init__()
        assert not (use_sigmoid and use_mask), "Cannot use both sigmoid and mask."
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        """
        Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor to average the loss.
            reduction_override (str, optional): The method to reduce the loss.
            ignore_index (int, optional): Override the default ignore_index.

        Returns:
            torch.Tensor: The calculated loss.
        """
        reduction = reduction_override if reduction_override else self.reduction
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index

        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        if self.use_sigmoid:
            # Binary Cross Entropy with Sigmoid for multi-label classification.
            loss = F.binary_cross_entropy_with_logits(
                cls_score, label.float(), weight=weight, reduction='none')
        elif self.use_mask:
            # Masked loss implementation here if needed.
            raise NotImplementedError("Mask loss is not implemented in this version.")
        else:
            # Standard cross-entropy loss for multi-class classification.
            loss = F.cross_entropy(
                cls_score, label, weight=class_weight, reduction='none', ignore_index=ignore_index)

        # Apply weight if provided
        if weight is not None:
            loss = loss * weight

        # Average or sum the loss
        if reduction == 'mean':
            if self.avg_non_ignore and ignore_index is not None:
                # Average over non-ignored elements
                loss = loss.sum() / (label != ignore_index).sum()
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        # Apply loss weight
        loss = loss * self.loss_weight

        return loss