import time

import torch
import torchvision.transforms as T
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import DTD
from tqdm import tqdm

from functional.utilities.data_utils import get_transform


class DescribableTexturesDataset(Dataset):
    def __init__(self, root: str = "dataset", split: str = "train", gray: bool = False, augment: bool = False):
        if augment:
            with open('config/datasets/augmentations_preset_1.yaml', 'r') as f:
                transforms_config = yaml.safe_load(f)
            transform = get_transform(transforms_config, img_size=224)
        else:
            transforms = [
                T.Resize(224),
                T.CenterCrop((224, 224)),
                T.ToTensor()
            ]
            if gray:
                transforms.append(T.Grayscale())
            transform = T.Compose(transforms)
        if split not in ["train", "test", "val"]:
            raise ValueError("Split must be either 'train', 'test', or 'val'")
        self.data = DTD(root=root,
                            transform=transform,
                            split=split,
                            download=True)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


def main():
    dataloader = torch.utils.data.DataLoader(DescribableTexturesDataset(split="train", augment=False),
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=True)
    _x = 0
    t0 = time.perf_counter()
    for (x, y) in tqdm(dataloader):
        print(x.shape, y)
        plt.imshow(x.squeeze(0).permute(1, 2, 0))
        plt.text(0, -12, str(dataloader.dataset.data.classes[y.item()]), color='green', fontsize=14, ha='left',
                 va='top')
        plt.show()
        time.sleep(1.5)
    print(f"{time.perf_counter() - t0:.2f} s")

if __name__ == "__main__":
    main()