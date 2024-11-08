"""This file contains a torch.utils.data.Dataset class for the MNIST-number dataset."""

import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Dataset class for MNIST-number dataset."""

    def __init__(
        self,
        train: bool = True,
        labels: None | list[int] = None,
        spurious_features: None | dict[int, callable] = None,
        probabilities: None | dict[int, float] = None,
        random_seed: int = None,
    ):
        """Download the MNIST Dataset (if not already downloaded). Filters for given labels and applies given spurious features.

        Args:
            train (bool, optional): True if this Dataset is used for training, else False. Defaults to True.
            labels (None | list[int], optional): Labels that should be used in the dataset. Should be a list of integers (corresponding labels are then used) or None (all labels are used). Defaults to None.
            spurious_features (None | dict[int, callable], optional): Contains all spurious functions that should be applied to a given label (key in dictionary). Defaults to None.
            probabilities (None | dict[int, float], optional): Contains the probabilities with which all specified spurious functions are applied to the corresponding label. Defaults to None.
            random_seed (int, optional): Seed for random initialization. Defaults to None.

        Raises:
            ValueError: If a specified spurious function has no corresponding probability.
        """
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
            missing_keys = set(spurious_features.keys()) - set(probabilities.keys())

            if missing_keys:
                raise ValueError(
                    f"The probabilities for the following keys are missing {missing_keys}."
                )

            for spurious_label in spurious_features:
                spurious_function = spurious_features[spurious_label]
                probability = probabilities[spurious_label]
                indices = [
                    i for i, (_, label) in enumerate(data) if label == spurious_label
                ]

                num_choices = int(len(indices) * probability)
                selected_indices = random.sample(indices, k=num_choices)

                self.spurious_indices[spurious_label] = selected_indices

                for i in selected_indices:
                    data[i] = (spurious_function(data[i][0]), data[i][1])

        self.data = data

    def __len__(self) -> int:
        """Return the lengths of the dataset.

        Returns:
            int: Length of Dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, bool]:
        """Return data item by index.

        Args:
            idx int: Index.

        Returns:
            tuple[torch.Tensor, int, int, bool]: Returns the data as tensor, the true and encoded label as integer and whether this data item includes a spurious feature.
        """
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
        """Plot a data item given a index.

        Args:
            idx int: Index.
        """
        image, label = self.data[idx]

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()

        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
