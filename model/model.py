import torch
from data_pre.data_preprocess import preprocess_image
from loss.parse_loss import parse_losses
from .basemodule import BaseModule
from .csp_darknet import YOLOv8CSPDarknet
from .optim import YOLOv5OptimizerConstructor
from .yolov8_head import YOLOv8Head
from .yolov8_pafpn import YOLOv8PAFPN


class Detector(BaseModule):
    def __init__(self,
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 last_stage_out_channels=512,
                 num_class=80,
                 lr=0.001,
                 weight_decay=0.0005,
                 batch_size=1,
                 ):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet(deepen_factor=deepen_factor, widen_factor=widen_factor, last_stage_out_channels=last_stage_out_channels)
        self.neck = YOLOv8PAFPN([256, 512, last_stage_out_channels], [256, 512, last_stage_out_channels], deepen_factor, widen_factor)
        self.bbox_head = YOLOv8Head(num_classes=80, in_channels=[256, 512, 512],
                                    train_cfg={'assigner': {'type': 'BatchTaskAlignedAssigner', 'num_classes': num_class, 'use_ciou': True, 'topk': 10, 'alpha': 0.5, 'beta': 6.0, 'eps': 1e-09}})
        optim_wrapper_config = {'type': 'OptimWrapper', 'clip_grad': {'max_norm': 10.0}, 'optimizer': {'type': 'SGD', 'lr': lr, 'momentum': 0.937, 'weight_decay': weight_decay, 'nesterov': True, 'batch_size_per_gpu': batch_size}}

        self.optim_wrapper = YOLOv5OptimizerConstructor(optim_wrapper_cfg=optim_wrapper_config,
                                                        paramwise_cfg=None)(self)
        self.init_weights()

    # def val(self, val_dataset):

    def save_model(self, ckp_dir):
        torch.save(self.state_dict(), ckp_dir)


    def step_train(self, x, data_sample):
        x = preprocess_image(x, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True)
        x = self.backbone(x)
        x = self.neck(x)
        losses = self.bbox_head.loss(x, data_sample)
        parsed_losses, log_vars = parse_losses(losses)
        self.optim_wrapper.update_params(parsed_losses)
        return log_vars