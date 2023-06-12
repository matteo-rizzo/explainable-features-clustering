import time
import yaml

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

from functional.data_utils import get_transform


class Food101Dataset(Dataset):
    def __init__(self, root: str = "dataset", train: bool = True, augment: bool = False):
        self.data = Food101(root=root, split="train" if train else "test", download=True)
        if augment:
            with open('config/datasets/augmentations_preset_1.yaml', 'r') as f:
                transforms_config = yaml.safe_load(f)
            self.transform = get_transform(transforms_config,(224, 224))
        else:
            self.transform = Compose([
                Resize((224, 224)),
                ToTensor()
            ])

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def main():
    dataloader = torch.utils.data.DataLoader(Food101Dataset(train=True, augment=True),
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=True)
    _x = 0
    t0 = time.perf_counter()
    for (x, y) in tqdm(dataloader):
        print(x.shape, y)
        plt.imshow(x.squeeze(0).permute(1, 2, 0))
        plt.show()
        # plt.clf()
        _x = _x + 1
        time.sleep(1)
    print(f"{time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
