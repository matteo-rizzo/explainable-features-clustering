from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm


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


def create_stratified_subset(dataset, num_samples):
    class_distribution = {}  # Dictionary to store class counts
    class_indices = {}  # Dictionary to store class indices
    num_classes = len(dataset.dataset.food101.classes)
    count = set()
    for i in range(num_classes):
        class_distribution[i] = 0
        class_indices[i] = []
    # Step 1: Calculate class distribution
    for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset)):
        label = label.item()
        if class_distribution[label] < num_samples:
            class_distribution[label] += 1
            class_indices[label].append(i)
        else:
            count.add(label)
        if len(count) == num_classes:
            break

    # Step 3: Initialize empty lists
    selected_indices = []
    for indices_list in class_indices.values():
        selected_indices.extend(indices_list)

    print(selected_indices)
    # Step 4: Create the subset dataset
    subset = Subset(dataset, selected_indices)

    return subset


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
