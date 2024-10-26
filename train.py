import torch
from datasets import Dtdatasets

class Args:
    def __init__(self):
        self.data_root = '<path to data_root>'
        self.isTrain = True
        self.batch_size = 32
        self.lr = 1e-4
        self.epoch = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = 'rgb' # 'red' 'rgb_red'

def get_parser():
    return Args()

if __name__ == '__main__':
    args = get_parser()
    datasets = Dtdatasets(args=args)
    dataloader = torch.utils.data.Dataloader(
        datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    print('img_number:', len(datasets))

    model = img2anchor(args)

    if args.mode == 'rgb':


    elif args.mode == 'red':


    elif args.mode == 'rgb_red':




