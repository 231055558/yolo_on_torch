import cv2
import torch
import torch.nn as nn

from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from model.basemodule import BaseModule
from model.csp_darknet import YOLOv8CSPDarknet
from model.yolov8_pafpn import YOLOv8PAFPN
from model.yolov8_head import YOLOv8Head


class DataSample:
    def __init__(self, metainfo=None):
        """
        自定义数据样本类，用于模拟 mmdet.structures 中的 DetDataSample 类。

        Args:
            metainfo (dict): 图像的元信息，默认为 None。
        """
        # 用于存储图像的元数据，比如原始尺寸、比例因子等
        self.metainfo = metainfo if metainfo is not None else {}

    def set_metainfo(self, key, value):
        """
        设置 metainfo 中的键值对。

        Args:
            key (str): 元信息的键名。
            value (any): 元信息的值。
        """
        self.metainfo[key] = value

    def get_metainfo(self, key, default=None):
        """
        获取 metainfo 中的值。

        Args:
            key (str): 元信息的键名。
            default (any): 如果键不存在，则返回的默认值。

        Returns:
            any: 对应的元信息值。
        """
        return self.metainfo.get(key, default)
# 假设你有一个模型类
class Detector(BaseModule):
    def __init__(self):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet()
        self.neck = YOLOv8PAFPN([256, 512, 512], [256, 512, 512])
        self.bbox_head = YOLOv8Head(80, [256, 512, 512])
    def forward(self, x):
        batch_size = x.shape[0]
        batch_data_samples = [DataSample() for _ in range(batch_size)]
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head.predict(x, batch_size_sample=batch_data_samples, rescale=True)
        return x

    def predict(self, x, data_sample):
        x = self.backbone(x)
        x = self.neck(x)
        result = self.bbox_head.predict(x, data_sample)
        return result

model = Detector()

ckp_path = './checkpoint/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
# # 假设权重文件路径为 'path_to_weights.pth'
checkpoint = torch.load('./checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth', map_location='cpu')

# 如果是直接保存的模型状态字典
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint



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
        else:
            print(f"{model_key}: Not found in weight file.")

    # 加载映射后的权重
    model.load_state_dict(new_state_dict, strict=False)
    return model

model = load_weights_with_mapping(model, ckp_path)

# 指定输入图片的路径
image_path = './demo/demo.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
transforms = []

transforms.append(LoadImageFromFile())
transforms.append(YOLOv5KeepRatioResize(scale=(1024, 1024)))
transforms.append(LetterResize(scale=(1024, 1024), allow_scale_up=False, pad_val={'img':114}))
transforms.append(LoadAnnotations(with_bbox=True))
transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
results = {"img_path": image_path, "img_id":0}
for t in transforms:
    results = t(results)
data_ = results
data_['inputs'] = [results['inputs']]
data_['data_samples'] = [data_['data_samples']]


# 进行预测
results_list = model.predict(data_['inputs'], data_['data_samples'])


# 转回到原图尺寸，提取bounding box坐标、置信度、类别
for box in results:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 转换为整数，得到左上和右下坐标
    confidence = box.conf[0]  # 获取置信度
    cls = int(box.cls[0])  # 获取类别索引
    label = model.names[cls]  # 获取类别名称

    # 在图片上绘制矩形框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 绘制标签
    cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 指定输出图片的路径
# output_image_path = 'path/to/your/output/image.jpg'  # 替换为你的输出路径
#
# # 保存图片
# cv2.imwrite(output_image_path, image)

# 显示结果图片
cv2.imshow('YOLO Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()