import cv2
import torch
import torch.nn as nn

from data_pre.data_preprocess import preprocess_image
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from data_pre.tolabel import load_txt_to_instance_data, parse_bbox_from_file
from model.basemodule import BaseModule
from model.csp_darknet import YOLOv8CSPDarknet
from model.yolov8_pafpn import YOLOv8PAFPN
from model.yolov8_head import YOLOv8Head
from utils import stack_batch, add_pred_to_datasample
import cv2
class Detector(BaseModule):
    def __init__(self):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet(deepen_factor=1.0, widen_factor=1.0, last_stage_out_channels=512)
        self.neck = YOLOv8PAFPN([256, 512, 512], [256, 512, 512], deepen_factor=1.0, widen_factor=1.0)
        head_cfg = {'multi_label': True, 'nms_pre': 30000, 'score_thr': 0.001,
                    'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'max_per_img': 300}
        self.bbox_head = YOLOv8Head(80, [256, 512, 512],
                                    train_cfg={'assigner': {'type': 'BatchTaskAlignedAssigner', 'num_classes': 80, 'use_ciou': True, 'topk': 10, 'alpha': 0.5, 'beta': 6.0, 'eps': 1e-09}})

    def forward(self, x, data_sample):
        x = preprocess_image(x, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True)
        x = self.backbone(x)
        x = self.neck(x)
        losses = self.bbox_head.loss(x, data_sample)
        return losses

def main():

    model = Detector()


    # 指定输入图片的路径
    image_path = './data/dota/train/images/P0000__1024__0___0.tiff'  # 替换为你的图片路径
    annfile_path = './data/dota/train/annfiles/P0000__1024__0___0.txt'
    image = cv2.imread(image_path)

    transforms = []

    transforms.append(LoadImageFromFile())
    transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
    transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}))
    transforms.append(LoadAnnotations(with_bbox=True))
    transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
    results = {"img_path": image_path, "img_id": 0}
    for t in transforms:
        results = t(results)
    batch_input = [results['inputs'].float()]
    data_ = results
    data_['inputs'] = stack_batch(batch_input, 1, 0)
    data_['data_samples'] = [data_['data_samples']]

    data_samples = {}
    data_samples['img_metas'] = [
        data_samples.metainfo for data_samples in data_['data_samples']
    ]
    instance_data = parse_bbox_from_file(annfile_path)

    data_samples['bboxes_labels'] = instance_data

    results = model(data_['inputs'], data_samples)


    print(results)


if __name__ == '__main__':
    main()
