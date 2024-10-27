import numpy as np
import torch.nn as nn
import torch

from typing import Dict, Optional, Tuple, Union, Sequence
import math
from functools import partial

from torch import Tensor
from torch import distributed as torch_dist

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')

    # 根据填充类型实例化相应的层
    if padding_type == 'zero':
        return nn.ZeroPad2d(*args, **kwargs, **cfg_)
    elif padding_type == 'reflect':
        return nn.ReflectionPad2d(*args, **kwargs, **cfg_)
    elif padding_type == 'replicate':
        return nn.ReplicationPad2d(*args, **kwargs, **cfg_)
    else:
        raise KeyError(f'Unsupported padding type: {padding_type}')

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # 定义类型到 PyTorch nn 模块的映射
    conv_layers = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'Conv': nn.Conv2d,  # 默认使用 Conv2d
    }

    # 根据类型获取相应的卷积层
    if layer_type not in conv_layers:
        raise KeyError(f'Unsupported convolution type: {layer_type}')

    conv_layer = conv_layers[layer_type]

    # 实例化卷积层并返回
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer

def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # 定义类型到 PyTorch nn 模块的映射
    norm_layers = {
        'BN': nn.BatchNorm2d,
        'SyncBN': nn.SyncBatchNorm,
        'GN': nn.GroupNorm,
        'IN': nn.InstanceNorm2d,
        'LN': nn.LayerNorm
    }

    # 根据类型获取相应的归一化层
    if layer_type not in norm_layers:
        raise KeyError(f'Unsupported normalization type: {layer_type}')

    norm_layer = norm_layers[layer_type]

    # 推断缩写形式
    abbr = layer_type.lower()

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    # 是否需要计算梯度
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    # 创建归一化层实例
    if layer_type == 'GN':
        if 'num_groups' not in cfg_:
            raise KeyError('The cfg dict must contain the key "num_groups" for GN')
        layer = norm_layer(num_channels=num_features, **cfg_)
    else:
        layer = norm_layer(num_features, **cfg_)

    # 设置参数的 requires_grad
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
class SiLU(nn.Module):
    """Sigmoid Weighted Liner Unit."""

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs) -> torch.Tensor:
        if self.inplace:
            return inputs.mul_(torch.sigmoid(inputs))
        else:
            return inputs * torch.sigmoid(inputs)

def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    layer_type = cfg.pop('type')

    # 激活层类型到 PyTorch 激活函数的映射
    activation_layers = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'RReLU': nn.RReLU,
        'ReLU6': nn.ReLU6,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'SiLU': nn.SiLU if torch.__version__ >= '1.7.0' else SiLU,
    }

    # 根据类型获取对应的激活层类
    if layer_type not in activation_layers:
        raise KeyError(f'Unsupported activation type: {layer_type}')

    activation_layer = activation_layers[layer_type]

    # 实例化激活层
    layer = activation_layer(**cfg)

    return layer

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def build_plugin_layer(cfg: Dict,
                       postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build a plugin layer in PyTorch.

    Args:
        cfg (dict): cfg should contain:
            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into abbreviation to create named layer.
            Default: ''.

    Returns:
        tuple[str, nn.Module]: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    # Extract the type of the layer and its arguments
    layer_type = cfg.pop('type')
    layer_args = cfg.copy()

    # Use getattr to obtain the desired layer class from nn or your custom layers
    # It assumes that layer_type is a valid PyTorch layer or a custom one
    if hasattr(nn, layer_type):
        plugin_layer = getattr(nn, layer_type)
    else:
        raise KeyError(f'Layer type {layer_type} is not found in torch.nn')

    # Create abbreviation based on layer type
    abbr = layer_type[:3].lower()  # Use the first three letters as abbreviation

    assert isinstance(postfix, (int, str)), "Postfix should be an integer or string."
    name = abbr + str(postfix)

    # Instantiate the layer with the provided arguments and any additional kwargs
    layer = plugin_layer(**layer_args, **kwargs)

    return name, layer

def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:

    assert isinstance(batch_gt_instances, Tensor)
    box_dim = batch_gt_instances.size(-1) - 2
    if len(batch_gt_instances) > 0:
        gt_images_indexes = batch_gt_instances[:, 0]
        max_gt_bbox_len = gt_images_indexes.unique(
            return_counts=True)[1].max()
        # fill zeros with length box_dim+1 if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance = torch.zeros(
            (batch_size, max_gt_bbox_len, box_dim + 1),
            dtype=batch_gt_instances.dtype,
            device=batch_gt_instances.device)

        for i in range(batch_size):
            match_indexes = gt_images_indexes == i
            gt_num = match_indexes.sum()
            if gt_num:
                batch_instance[i, :gt_num] = batch_gt_instances[
                    match_indexes, 1:]
    else:
        batch_instance = torch.zeros((batch_size, 0, box_dim + 1),
                                     dtype=batch_gt_instances.dtype,
                                     device=batch_gt_instances.device)

    return batch_instance

def get_dist_info(group = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size

def get_world_size(group = None) -> int:
    # if is_distributed():
    #     # handle low versions of torch like 1.5.0 which does not support
    #     # passing in None for group argument
    #     if group is None:
    #         group = get_default_group()
    #     return torch_dist.get_world_size(group)
    # else:
    #     return 1
    return 1

def get_rank(group = None) -> int:
    return 0

def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()

def filter_scores_and_topk(
        scores: torch.Tensor,
        score_thr: float,
        nms_pre: int,
        results: Dict[str, torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    过滤掉低于置信度阈值的检测结果，并保留前 k 个分数最高的结果。

    Args:
        scores (Tensor): 形状为 (num_instances, 1) 的置信度分数。
        score_thr (float): 置信度阈值，低于该值的检测结果会被过滤。
        nms_pre (int): 保留的前 n 个高置信度结果。
        results (dict, optional): 其他检测结果的字典，比如 labels。默认为 None。

    Returns:
        scores (Tensor): 过滤后的分数，形状为 (num_filtered_instances,).
        labels (Tensor): 对应的类别标签。
        keep_idxs (Tensor): 保留的结果的索引。
        results (dict): 更新后的其他检测结果字典。
    """
    # 根据置信度阈值筛选分数
    valid_mask = scores > score_thr
    valid_scores = scores[valid_mask]

    if results is not None:
        for key, value in results.items():
            results[key] = value[valid_mask]

    # 如果过滤后结果数大于 nms_pre，取前 nms_pre 个
    if len(valid_scores) > nms_pre:
        topk_scores, keep_idxs = valid_scores.topk(nms_pre, sorted=False)
        valid_scores = topk_scores
        if results is not None:
            for key, value in results.items():
                results[key] = value[keep_idxs]
    else:
        keep_idxs = torch.nonzero(valid_mask, as_tuple=True)[0]

    return valid_scores, keep_idxs, results

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def is_cuda_available() -> bool:
    """Returns True if cuda devices exist."""
    return torch.cuda.is_available()

def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | npu | mlu | mps | musa | cpu.
    """
    DEVICE = 'cpu'
    if is_cuda_available():
        DEVICE = 'cuda'
    return DEVICE

