import torch.nn as nn
import torch

from typing import Dict, Optional, Tuple, Union
import math
from functools import partial


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