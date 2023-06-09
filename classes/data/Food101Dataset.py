from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm


class Food101Dataset(Dataset):
    def __init__(self, root: str = "dataset", train: bool = True):
        self.data = Food101(root=root, split="train" if train else "test", download=True)
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
    pass
    # _x = 0
    # t0 = time.perf_counter()
    # for (x, y) in tqdm(dataloader):
    #     print(x.shape, y)
    #     plt.imshow(x.squeeze(0).permute(1, 2, 0))
    #     plt.show()
    #     # plt.clf()
    #     _x = _x + 1
    # print(f"{time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
