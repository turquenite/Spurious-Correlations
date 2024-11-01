"""This file serves as a playground for experimenting with various classes and models in the machine learning pipeline."""

from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model_trainer import train
from models import SimpleModel
from spurious_features import Position, spurious_square

if __name__ == "__main__":
    train_dataset = MNISTDataset(
        train=True,
        labels=[9, 7],
        spurious_features={
            9: [
                lambda img: spurious_square(img, pos=Position.LEFT_TOP, size=4),
                0.8,
            ]
        },
    )
    validation_dataset = MNISTDataset(train=False, labels=[9, 7])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    model = SimpleModel(num_classes=2)

    train(
        model=model,
        validation_loader=validation_loader,
        train_loader=train_loader,
        num_epochs=10,
    )
