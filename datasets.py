import os # 内置库
# import cv2 # 和PIL二选一
from PIL import Image # 需要安装

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class Dtdatasets():
    def __init__(self, args):
        self.data_root = args.data_root

        self.load_size = 1024
        self.crop_size = 256

        self.rgb_nc = 3
        self.red_nc = 1

        self.dir = os.path.join(self.data_root, 'train')
        self.mode = args.mode

        self.dir_rgb = sorted(make_dataset(os.path.join(self.dir, 'rgb')))
        self.dir_red = sorted(make_dataset(os.path.join(self.dir, 'red')))
        # self.dir_rgb_red = sorted(make_dataset(os.path.join(self.dir, 'rgb_red')))

    def __getitem__(self, index):
        if self.mode=='rgb':
            img_path = self.dir_rgb[index]
            img = Image.open(img_path).convert('RGB')
            return {'img': img}
        elif self.mode=='red':
            img_path = self.dir_red[index]
            img = Image.open(img_path).convert('RGB')
            return {'img': img}
        elif self.mode=='rgb_red':
            rgb_path = self.dir_rgb[index]
            red_path = self.dir_red[index]
            rgb_img = Image.open(rgb_path).convert('RGB')
            red_img = Image.open(red_path).convert('RGB')
            return {'rgb': rgb_img, 'reb': red_img}

    def __len__(self):
        if self.mode=='rgb':
            return len(self.dir_rgb)
        elif self.mode=='red':
            return len(self.dir_red)
        elif self.mode=='rgb_red':
            return len(self.dir_rgb) + len(self.dir_red)




