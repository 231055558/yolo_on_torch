import torch
from tqdm import tqdm

from dataset import Ydataset
from model.model import Detector
from transform import Transform
from torch.utils.data import DataLoader
def collate_fn(batch):
    images, targets = zip(*batch)  # 分离图像和目标
    images = torch.stack(images, dim=0)  # 处理图像

    # 处理目标
    max_length = max(len(t) for t in targets)  # 找到最大长度
    padded_targets = []
    for target in targets:
        if len(target) < max_length:
            # 用零填充
            padded_target = torch.cat([target, torch.zeros((max_length - len(target), target.size(1)))])
        else:
            padded_target = target
        padded_targets.append(padded_target)
    padded_targets = torch.stack(padded_targets, dim=0)
    return images, padded_targets
def train_model(dataset, num_epochs, batch_size, device, ckp_dir):
    yolo_model = Detector(batch_size=batch_size).to(device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        yolo_model.train()
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for img, labels in tepoch:
                img = img.to(device)
                labels = labels.to(device)
                data_sample = {'bboxes_labels': labels, 'img_metas': labels}
                losses = yolo_model.step_train(img, data_sample)
                print(losses)
    yolo_model.save_model(ckp_dir)




if __name__ == '__main__':
    img_dir = './data/coco/val2017/'
    label_dir = './data/coco/annfiles/'
    ckp_dir = './checkpoints/oct30.pth'
    dataset = Ydataset(img_dir, label_dir, transform=True)
    train_model(dataset, num_epochs=20, batch_size=8, device=torch.device('cpu'),ckp_dir=ckp_dir)
