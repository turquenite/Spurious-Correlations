from torch.utils.data import DataLoader

from dataset import MNISTDataset
from spurious_features import Orientation, spurious_lines

if __name__ == "__main__":
    # Create instances for training and testing datasets
    train_dataset = MNISTDataset(
        train=True,
        labels=[9, 7],
        spurious_features={
            9: [
                lambda img: spurious_lines(
                    img, orientation=Orientation.VERTICAL, distance=7
                ),
                1,
            ]
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
