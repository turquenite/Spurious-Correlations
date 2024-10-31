import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        labels: None | list[int] = None,
        spurious_features: None | dict[int : list[callable, float]] = None,
        random_seed: int = None,
    ):
        self.train = train
        self.spurious_features = spurious_features
        self.spurious_indices = {}
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if random_seed:
            random.seed(random_seed)

        full_dataset = datasets.MNIST(
            root="mnist_data", train=self.train, download=True, transform=self.transform
        )

        if labels:
            self.label_encoding = {label: idx for idx, label in enumerate(labels)}
            indices = [
                i for i, (_, label) in enumerate(full_dataset) if label in labels
            ]
            data_subset = torch.utils.data.Subset(full_dataset, indices)
            data = [(img, label) for img, label in data_subset]

        else:
            self.label_encoding = {
                label: idx for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            }
            data = [(img, label) for img, label in full_dataset]

        if spurious_features:
            for spurious_label in spurious_features:
                spurious_function, probability = spurious_features[spurious_label]
                indices = [
                    i for i, (_, label) in enumerate(data) if label == spurious_label
                ]

                num_choices = int(len(indices) * probability)
                selected_indices = random.choices(indices, k=num_choices)

                self.spurious_indices[spurious_label] = selected_indices

                for i in selected_indices:
                    data[i] = (spurious_function(data[i][0]), data[i][1])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, true_label = self.data[idx]
        encoded_label = self.label_encoding[true_label]
        return (
            data,
            true_label,
            encoded_label,
            True
            if true_label in self.spurious_indices
            and idx in self.spurious_indices[true_label]
            else False,
        )

    def view_item(self, idx):
        image, label = self.data[idx]

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()

        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
