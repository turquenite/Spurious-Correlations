"""This file contains a ML train loop for the MNIST-number dataset."""

import os
from collections import defaultdict
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(
    num_epochs: int,
    model: torch.nn.Module,
    validation_loader: DataLoader,
    train_loader: DataLoader,
    optimizer_type=torch.optim.Adam,
    lr: float = 0.001,
    loss_function=torch.nn.CrossEntropyLoss(),
):
    """Trains given model for num_epochs.

    Args:
        num_epochs (int): Number of epochs.
        model (torch.nn.Module): Model architecture.
        validation_loader (DataLoader): DataLoader for validation set.
        train_loader (DataLoader): DataLoader for train set.
        optimizer_type (_type_, optional): Optimizer used during training. Defaults to torch.optim.Adam.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        loss_function (_type_, optional): Loss function. Defaults to torch.nn.CrossEntropyLoss().
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/spurious_trainer_{}".format(timestamp))
    epoch_number = 0

    optimizer = optimizer_type(model.parameters(), lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    best_vloss = 1_000_000.0

    # Add sample plots in Tensorboard
    example_batch = iter(train_loader)
    example_plots, example_true_label, _, _ = next(example_batch)

    img_grid = torchvision.utils.make_grid(example_plots)

    writer.add_image("MNIST Samples", img_grid)

    for _ in range(num_epochs):
        print("EPOCH {}:".format(epoch_number + 1))

        # Train Mode
        model.train(True)
        avg_loss, avg_accuracy, avg_worst_group_accuracy = _train_one_epoch(
            epoch_number, writer, model, train_loader, loss_function, optimizer, device
        )

        running_vloss = 0.0
        num_predictions = defaultdict(int)
        num_correct_predictions = defaultdict(int)

        # Evaluation Mode
        model.eval()

        # Disable gradient computation
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vtrue_labels, vencoded_labels, vspurious = vdata
                vinputs, vtrue_labels, vencoded_labels, vspurious = (
                    vinputs.to(device),
                    vtrue_labels.to(device),
                    vencoded_labels.to(device),
                    vspurious.to(device),
                )
                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vencoded_labels)
                running_vloss += vloss.item()

                _, predicted_labels = torch.max(voutputs, 1)

                for pred, encoded_label, true_label, spur in zip(
                    predicted_labels, vencoded_labels, vtrue_labels, vspurious
                ):
                    key = f"{true_label} - {'spurious' if spur else 'not spurious'}"

                    if pred.item() == encoded_label.item():
                        num_correct_predictions[key] += 1

                    num_predictions[key] += 1

        avg_vloss = float(running_vloss / (i + 1))
        group_vaccuracies = {
            group: num_correct_predictions[group] / num_predictions[group]
            for group in num_predictions
        }
        avg_worst_group_vaccuracy = min(group_vaccuracies.values())

        total_correct = sum(num_correct_predictions.values())
        total_predictions = sum(num_predictions.values())
        avg_vaccuracy = (
            total_correct / total_predictions if total_predictions > 0 else 0
        )

        print(
            "LOSS train {} valid {} | ACCURACY train {} valid {} | WORST GROUP ACCURACY {}".format(
                avg_loss,
                avg_vloss,
                avg_accuracy,
                avg_vaccuracy,
                avg_worst_group_vaccuracy,
            )
        )

        # Log the running loss and accuracy averaged per batch
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )

        writer.add_scalars(
            "Accuracy",
            {"Training": avg_accuracy, "Validation": avg_vaccuracy},
            epoch_number + 1,
        )

        writer.add_scalars(
            "Worst Group Accuracy",
            {
                "Training": avg_worst_group_accuracy,
                "Validation": avg_worst_group_vaccuracy,
            },
            epoch_number + 1,
        )

        for group, accuracy in group_vaccuracies.items():
            writer.add_scalar(f"Accuracy/{group}/valid", accuracy, epoch_number + 1)

        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs("models", exist_ok=True)
            model_path = r"models\model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def _train_one_epoch(
    current_epoch: int,
    tb_writer: SummaryWriter,
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function,
    optimizer,
    device,
):
    running_loss = 0.0
    last_loss = 0.0
    num_predictions = defaultdict(int)
    num_correct_predictions = defaultdict(int)

    N = len(train_loader) // 2 - 1

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, true_labels, encoded_labels, spurious = data

        inputs, true_labels, encoded_labels, spurious = (
            inputs.to(device),
            true_labels.to(device),
            encoded_labels.to(device),
            spurious.to(device),
        )

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Fetch predicted labels for computing accuracy labels later
        _, predicted_labels = torch.max(outputs, 1)

        for pred, encoded_label, true_label, spur in zip(
            predicted_labels, encoded_labels, true_labels, spurious
        ):
            key = f"{true_label} - {'spurious' if spur else 'not spurious'}"

            if pred.item() == encoded_label.item():
                num_correct_predictions[key] += 1

            num_predictions[key] += 1

        # Compute the loss and its gradients
        loss = loss_function(outputs, encoded_labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % N == N - 1:
            last_loss = running_loss / N

            group_accuracies = {
                group: num_correct_predictions[group] / num_predictions[group]
                for group in num_predictions
            }
            worst_group_accuracy = min(group_accuracies.values())

            total_correct = sum(num_correct_predictions.values())
            total_predictions = sum(num_predictions.values())
            overall_accuracy = (
                total_correct / total_predictions if total_predictions > 0 else 0
            )

            print(
                "  batch {} loss: {} accuracy: {} worst group accuracy".format(
                    i + 1,
                    last_loss,
                    overall_accuracy,
                )
            )

            tb_x = current_epoch * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)

            tb_writer.add_scalar("Accuracy/train", overall_accuracy, tb_x)

            for group, accuracy in group_accuracies.items():
                tb_writer.add_scalar(f"Accuracy/{group}/train", accuracy, tb_x)

            tb_writer.add_scalar(
                "Accuracy/worst_group/train", worst_group_accuracy, tb_x
            )

            running_loss = 0.0

    avg_loss = running_loss / len(train_loader)

    group_accuracies = {
        group: num_correct_predictions[group] / num_predictions[group]
        for group in num_predictions
    }
    worst_group_accuracy = min(group_accuracies.values())

    total_correct = sum(num_correct_predictions.values())
    total_predictions = sum(num_predictions.values())
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    return avg_loss, overall_accuracy, worst_group_accuracy
