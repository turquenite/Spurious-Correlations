"""This file contains a torch.utils.data.Dataset class for the MNIST-number dataset."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from spurious_features import SpuriousFeature


class MNISTDataset(Dataset):
    """Dataset class for MNIST-number dataset."""

    def __init__(
        self,
        train: bool = True,
        labels: None | list[int] = None,
        main_spurious_features: None | SpuriousFeature = None,
        minority_spurious_features: None | SpuriousFeature = None,
        probabilities: None | dict[int, float] = None,
        random_seed: int = None,
    ):
        """Download the MNIST Dataset (if not already downloaded). Filters for given labels and applies given spurious features.

        Args:
            train (bool, optional): True if this Dataset is used for training, else False. Defaults to True.
            labels (None | list[int], optional): Labels that should be used in the dataset. Should be a list of integers (corresponding labels are then used) or None (all labels are used). Defaults to None.
            main_spurious_features (None | dict[int, SpuriousFeature], optional): Contains all spurious features that should be applied to a given label (key in dictionary) with a given probability. Defaults to None.
            minority_spurious_features (None | dict[int, SpuriousFeature], optional): Contains all spurious features that should be applied all samples of a given label (key in dict), where no main_spurious_feature was applied. Defaults to None.
            probabilities (None | dict[int, float], optional): Contains the probabilities with which all specified spurious features in main_spurious_features are applied to the corresponding label. Defaults to None.
            random_seed (int, optional): Seed for random initialization. Defaults to None.

        Raises:
            ValueError: If a specified spurious function has no corresponding probability.
        """
        self.train = train
        self.main_spurious_features = main_spurious_features
        self.minority_spurious_features = minority_spurious_features
        self.main_spurious_indices = {}
        self.minority_spurious_indices = {}

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

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

        self.reverse_label_encoding = {v: k for k, v in self.label_encoding.items()}

        if main_spurious_features:
            missing_keys = set(main_spurious_features.keys()) - set(
                probabilities.keys()
            )

            if missing_keys:
                raise ValueError(
                    f"The probabilities for the following keys are missing {missing_keys}."
                )

            for spurious_label, spurious_feature in main_spurious_features.items():
                probability = probabilities[spurious_label]
                indices = [
                    i for i, (_, label) in enumerate(data) if label == spurious_label
                ]

                num_choices = int(len(indices) * probability)
                selected_indices = random.sample(indices, k=num_choices)

                self.main_spurious_indices[spurious_label] = selected_indices

                for i in selected_indices:
                    data[i] = (spurious_feature.apply(data[i][0]), data[i][1])

                if (
                    minority_spurious_features
                    and spurious_label in minority_spurious_features
                ):
                    minority_spurious_feature = minority_spurious_features[
                        spurious_label
                    ]

                    self.minority_spurious_indices[spurious_label] = list(
                        set(indices) - set(selected_indices)
                    )

                    for i in self.minority_spurious_indices[spurious_label]:
                        data[i] = (
                            minority_spurious_feature.apply(data[i][0]),
                            data[i][1],
                        )

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
            tuple[torch.Tensor, int, int, str]: Returns the data as tensor, the true and encoded label as integer and a short description whether this data item includes a spurious feature.
        """
        data, true_label = self.data[idx]
        encoded_label = self.label_encoding[true_label]

        if (
            self.main_spurious_features
            and true_label in self.main_spurious_features
            and idx in self.main_spurious_indices[true_label]
        ):
            description = self.main_spurious_features[true_label].description

        elif (
            self.minority_spurious_features
            and true_label in self.minority_spurious_features
            and idx in self.minority_spurious_indices[true_label]
        ):
            description = self.minority_spurious_features[true_label].description

        else:
            description = "Not Spurious"

        return (
            data,
            true_label,
            encoded_label,
            description,
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

    def gradcam(self, model: torch.nn.Module, idx: int, target_layer: torch.nn.Module):
        """Visualize the importance of each pixel for one data-sample in a target_layer.

        Args:
            model (torch.nn.Module): Model to evaluate.
            idx (int): Index of data sample.
            target_layer (torch.nn.Module): Layer of model used for computing gradients.
        """
        model.cpu().eval()

        data, true_label = self.data[idx]
        data = data.unsqueeze(0).requires_grad_()

        activations = []
        gradients = []

        # Forward hook for activations
        def forward_hook(module, input, output):
            activations.append(output)

        # Backward hook for gradients
        def full_backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(full_backward_hook)

        # Forward pass
        output = model(data)
        pred_label = output.argmax(dim=1).item()

        # Backward pass
        model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, pred_label] = 1
        output.backward(gradient=one_hot_output)

        gradients = gradients[0].cpu().detach()
        activations = activations[0].cpu().detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        gradcam_map = (weights * activations).sum(dim=1).squeeze()

        # Normalization
        gradcam_map = gradcam_map.clamp(min=0)
        gradcam_map = (gradcam_map - gradcam_map.min()) / (
            gradcam_map.max() - gradcam_map.min()
        )
        gradcam_map = (
            interpolate(
                gradcam_map.unsqueeze(0).unsqueeze(0),
                size=data.shape[2:],
                mode="bilinear",
            )
            .squeeze()
            .numpy()
        )

        original_image = data.squeeze().cpu().detach().numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image (Label: {true_label})")
        plt.imshow(original_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(
            f"Grad-CAM Heatmap (Predicted Label: {self.reverse_label_encoding[pred_label]})"
        )
        plt.imshow(original_image, cmap="gray", alpha=0.5)
        plt.imshow(gradcam_map, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.show()

        forward_handle.remove()
        backward_handle.remove()
