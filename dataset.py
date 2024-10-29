import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
class Ydataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
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

    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取索引idx对应的样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label_path = self.samples[idx]

        # 加载图片
        image = Image.open(image_path).convert('RGB')

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    x1, y1, x2, y2, label = map(float, parts)
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(label)

        # 如果有预处理/转换操作，应用在这里
        if self.transform:
            image = self.transform(image)

        # 将边界框和标签转换为Tensor
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 构建目标字典
        target = {
            'boxes': bboxes,
            'labels': labels
        }

        return image, target