from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.MNISTDataset import MNISTDataset
from classes.ModelImportanceWeightedCNN import ModelImportanceWeightedCNN
from functional.setup import get_device

DEVICE_TYPE = "cpu"
OPTIMIZER = "sgd"
LEARNING_RATE = 0.01
CRITERION = "CrossEntropyLoss"
EPOCHS = 15


def main():
    train_loader = DataLoader(MNISTDataset(train=True), batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(MNISTDataset(train=False), batch_size=64, shuffle=False, num_workers=2)

    device = get_device(DEVICE_TYPE)

    model = ModelImportanceWeightedCNN(device)
    model.set_optimizer(OPTIMIZER, LEARNING_RATE)
    model.set_criterion(CRITERION)
    model.train_mode()

    for epoch in range(EPOCHS):

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y, _) in tqdm(enumerate(train_loader), desc="Training epoch: {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            o = model.predict(x).to(device)
            loss = model.update_weights(o, y)
            running_loss += loss
            total, correct = model.get_accuracy(o, y, total, correct)

        train_loss, train_accuracy = running_loss / len(train_loader), 100 * correct / total

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y, _) in tqdm(enumerate(test_loader), desc="Testing epoch: {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            o = model.predict(x).to(device)
            loss = model.get_loss(o, y)
            running_loss += loss
            total, correct = model.get_accuracy(o, y, total, correct)

        test_loss, test_accuracy = running_loss / len(test_loader), 100 * correct / total

        print('Epoch [%d], train loss: %.3f, train accuracy: %.3f, test loss: %.3f, test accuracy: %.3f' % (
            epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy))


if __name__ == "__main__":
    main()
