from dataset import Ydataset

def train():



if __name__ == '__main__':
    img_dir = './data/coco/val2017/'
    label_dir = './data/coco/annfiles/'
    YOLO_Dataset = Ydataset(img_dir, label_dir, )