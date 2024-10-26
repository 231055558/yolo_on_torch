import torch
import torch.nn as nn

from model.basemodule import BaseModule
from model.csp_darknet import YOLOv8CSPDarknet
from model.yolov8_pafpn import YOLOv8PAFPN
# 假设你有一个模型类
class Detector(BaseModule):
    def __init__(self):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet()
        self.neck = YOLOv8PAFPN([256, 512, 512], [256, 512, 512])

model = Detector()
# neck = YOLOv8PAFPN([256, 512, 1024], [256, 512, 1024])  # 实例化模型
#
# # 打印模型结构
# print(model)
# print('\n\n\n\n------------------------------------------\n\n\n\n')
# # 更详细地打印每一层
for name, layer in model.named_modules():
    print(f"Layer: {name} \n Structure: {layer}\n\n")
ckp_path = './checkpoint/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
# # 假设权重文件路径为 'path_to_weights.pth'
checkpoint = torch.load('./checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth', map_location='cpu')
#
# # 查看权重文件中的键
# print(checkpoint.keys())
#
# 如果是直接保存的模型状态字典
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
#
# 查看每个权重的名称和形状
for name, weight in state_dict.items():
    print(f"Weight: {name}, Shape: {weight.shape}")

# 模型参数字典
model_state_dict = model.state_dict()

# # 比较两者的键和形状
# for name, param in model_state_dict.items():
#     if name in state_dict:
#         if param.shape == state_dict[name].shape:
#             print(f"{name}: Shape matches.")
#         else:
#             print(f"{name}: Shape mismatch! Model shape: {param.shape}, Weight shape: {state_dict[name].shape}")
#     else:
#         print(f"{name}: Not found in weight file.")


def load_weights_with_mapping(model, weight_path):
    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 创建一个新的字典用于存储映射后的权重
    new_state_dict = {}
    for model_key in model.state_dict().keys():
        # 获取权重文件中对应的键名
        # checkpoint_key = model_key.replace('stem.', 'backbone.stem.')

        if model_key in model_weights:
            new_state_dict[model_key] = model_weights[model_key]
            print(f'{model_key} is ok')
        else:
            print(f"{model_key}: Not found in weight file.")

    # 加载映射后的权重
    model.load_state_dict(new_state_dict, strict=False)
    return model

load_weights_with_mapping(model, ckp_path)