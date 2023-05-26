import time

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor, Resize, Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

class Food101Dataset(Dataset):
    def __init__(self, root: str = "dataset", train: bool = True):
        self.food101 = Food101(root=root, split="train" if train else "test", download=True)
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor()
        ])
    def __getitem__(self, index):
        img, label = self.food101[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.food101)


if __name__ == "__main__":
    dataset = Food101Dataset(train=True)
    dataloader = DataLoader(dataset, shuffle=True)

    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=100 * 10)

    # Warp into Subsets and DataLoaders
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=10)
    test_loader = DataLoader(train_dataset, shuffle=False, num_workers=2, batch_size=10)


    _x = 0
    t0 = time.perf_counter()
    for (x, y) in tqdm(dataloader):
        print(x.shape, y)
        plt.imshow(x.squeeze(0).permute(1, 2, 0))
        plt.show()
        # plt.clf()
        _x = _x + 1
    print(f"{time.perf_counter() - t0:.2f} s")
