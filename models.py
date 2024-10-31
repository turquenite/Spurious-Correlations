import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleModel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # output: 32 x 28 x 28
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # output: 64 x 28 x 28
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )  # output: 64 x 7 x 7

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened image size: 64 * 7 * 7
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional Layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x28x28 -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 64x14x14 -> 64x7x7

        x = x.view(-1, 64 * 7 * 7)  # Flatten 64 x 7 x 7 into 3136

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
