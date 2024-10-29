from data_pre.data_preprocess import preprocess_image
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from data_pre.tolabel import parse_bbox_from_file
from loss.parse_loss import parse_losses
from model.basemodule import BaseModule
from model.csp_darknet import YOLOv8CSPDarknet
from model.optim import YOLOv5OptimizerConstructor
from model.yolov8_pafpn import YOLOv8PAFPN
from model.yolov8_head import YOLOv8Head
from utils import stack_batch
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

    def set_train(self):
        optim_wrapper_config = {'type': 'OptimWrapper', 'clip_grad': {'max_norm': 10.0}, 'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'nesterov': True, 'batch_size_per_gpu': 4}}

        self.optim_wrapper = YOLOv5OptimizerConstructor(optim_wrapper_cfg=optim_wrapper_config,
                                                        paramwise_cfg=None)(self)


    def forward(self, x, data_sample):
        x = preprocess_image(x, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True)
        x = self.backbone(x)
        x = self.neck(x)
        losses = self.bbox_head.loss(x, data_sample)
        parsed_losses, log_vars = parse_losses(losses)
        self.optim_wrapper.update_params(parsed_losses)
        return log_vars

def main():

    model = Detector()
    model.set_train()


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
