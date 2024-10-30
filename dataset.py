import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
def transform_labels(labels_tensor, ori_shape, img_shape):
    # Unpack original and target dimensions
    ori_h, ori_w = ori_shape
    img_h, img_w = img_shape[:2]

    # 1. Calculate scale factors for aspect-ratio-preserving resize
    scale_x = img_w / ori_w
    scale_y = img_h / ori_h
    scale = min(scale_x, scale_y)  # Choose the smallest to keep aspect ratio

    # 2. Apply the scaling to all bounding box coordinates
    labels_tensor[:, 2] *= scale  # x1
    labels_tensor[:, 3] *= scale  # y1
    labels_tensor[:, 4] *= scale  # x2
    labels_tensor[:, 5] *= scale  # y2

    # 3. Calculate padding if the aspect ratio doesn't match
    pad_x = (img_w - ori_w * scale) / 2
    pad_y = (img_h - ori_h * scale) / 2

    # 4. Adjust for padding by adding offsets
    labels_tensor[:, 2] += pad_x  # x1
    labels_tensor[:, 3] += pad_y  # y1
    labels_tensor[:, 4] += pad_x  # x2
    labels_tensor[:, 5] += pad_y  # y2

    return labels_tensor

class Ydataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None):
        """
        Args:
            image_dir (string): 包含所有图片的目录路径。
            label_dir (string): 包含所有标签文件的目录路径。
            transform (callable, optional): 可选的transform函数，应用于图片上。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        # 获取所有图片文件名
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.samples.append((image_path, label_path))

        if transform:
            self.transforms = []
            self.transforms.append(LoadImageFromFile())
            self.transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
            self.transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}))
            self.transforms.append(LoadAnnotations(with_bbox=True))
            self.transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
        else:
            self.transforms = None


    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取索引idx对应的样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label_path = self.samples[idx]
        img_data = {"img_path": image_path, "img_id": idx}
        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                x1, y1, x2, y2, label, _ = map(float, parts)
                bboxes.append([idx, label, x1, y1, x2, y2])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        for t in self.transforms:
            results = t(img_data)

        img = results['inputs']
        data_sample = results['data_samples'].metainfo
        bboxes = transform_labels(bboxes, data_sample['ori_shape'], data_sample['img_shape'])

        return img, bboxes