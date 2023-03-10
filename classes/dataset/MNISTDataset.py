import os

import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

"""
Original dataset: https://yann.lecun.com/exdb/mnist/
CSV version: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
PNG version: https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz
"""

PATH_TO_DATASET = os.path.join("dataset", "mnist", "images")


class MNISTDataset(Dataset):

    def __init__(self, train: bool = True):
        super(Dataset, self).__init__()
        path_to_labels = os.path.join(PATH_TO_DATASET, "training" if train else "testing")
        labels = os.listdir(path_to_labels)

        self.__paths_to_data = []
        for label in labels:
            if label == ".DS_Store":
                continue
            path_to_label = os.path.join(path_to_labels, label)
            for img in os.listdir(path_to_label):
                if img == ".DS_Store":
                    continue
                self.__paths_to_data.append(os.path.join(path_to_label, img))

    def __getitem__(self, index) -> tuple[Tensor, Tensor, str]:
        path_to_img = self.__paths_to_data[index]
        label = Tensor([int(path_to_img.split(os.sep)[-2])])
        img = Tensor(read_image(path_to_img))
        return img, label, path_to_img

    def __len__(self):
        return len(self.__paths_to_data)


if __name__ == "__main__":
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)

    for (x, y, z) in dataloader:
        print(z, x.shape, y)
        plt.imshow(x.squeeze(0).permute(1, 2, 0))
        plt.show()
        plt.clf()
