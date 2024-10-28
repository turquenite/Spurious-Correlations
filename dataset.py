import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from spurious_features import Position, spurious_square


class MNISTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        labels: None | list[int] = None,
        spurious_features: None | dict[int : list[callable, float]] = None,
        *args,
        **kwargs,
    ):
        self.train = train
        self.spurious_features = spurious_features
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        full_dataset = datasets.MNIST(
            root="mnist_data", train=self.train, download=True, transform=self.transform
        )

        if labels:
            indices = [
                i for i, (_, label) in enumerate(full_dataset) if label in labels
            ]
            self.data = torch.utils.data.Subset(full_dataset, indices)

        else:
            self.data = full_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if label in self.spurious_features:
            image = self.spurious_features[label](image)

        return image, label

    def view_item(self, idx):
        image, label = self.data[idx]

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()

        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Create instances for training and testing datasets
    train_dataset = MNISTDataset(
        train=True,
        labels=[9, 7],
        spurious_features={
            9: lambda img: spurious_square(img, pos=Position.RIGHT_TOP, size=3)
        },
    )
    test_dataset = MNISTDataset(train=False, labels=[0, 7])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Test iterating over a batch
    for images, labels in train_loader:
        print(f"Batch of images has shape: {images.shape}")
        print(f"Batch of labels has shape: {labels.shape}")
        break

    train_dataset.view_item(3)
