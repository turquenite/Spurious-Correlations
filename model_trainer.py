"""This file contains a ML train loop for the MNIST-number dataset."""

import math
import os
from collections import defaultdict
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    num_epochs: int,
    model: torch.nn.Module,
    validation_loaders: dict[str, DataLoader],
    train_loader: DataLoader,
    optimizer_type=torch.optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0,
    loss_function=torch.nn.CrossEntropyLoss(),
    use_early_stopping: bool = True,
    patience: int = 5,
    experiment_description: str | None = None,
) -> tuple[str, str]:
    """Trains given model for num_epochs.

    Args:
        num_epochs (int): Number of epochs.
        model (torch.nn.Module): Model architecture.
        validation_loaders (dict[str, DataLoader]): A dictionary containing all Dataloader which should be used for evaluation. The key should serve as description.
        train_loader (DataLoader): DataLoader for train set.
        optimizer_type (_type_, optional): Optimizer used during training. Defaults to torch.optim.Adam.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.
        loss_function (_type_, optional): Loss function. Defaults to torch.nn.CrossEntropyLoss().
        use_early_stopping (bool, optional): Whether to use (True) or not use (False) early stopping. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement before triggering early stopping. Defaults to 5.
        experiment_description: str|None: Short experiment description used for saving and logging if not None.

    Returns:
        tuple: (str, str) containing:
            - Path to the saved model with the best validation loss.
            - Path to the TensorBoard log directory.
    """
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")

    tensorboard_log_dir_path = (
        f"runs/{timestamp}_{experiment_description}"
        if experiment_description
        else f"runs/{timestamp}"
    )

    writer = SummaryWriter(tensorboard_log_dir_path)

    early_stopping_counter = 0

    optimizer = optimizer_type(model.parameters(), lr, weight_decay=weight_decay)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    best_vloss = math.inf
    model_path = None

    # Add sample plots in Tensorboard
    example_batch = iter(train_loader)
    example_plots, example_true_label, _, _ = next(example_batch)

    img_grid = torchvision.utils.make_grid(example_plots)
    writer.add_image("Samples/MNIST", img_grid)

    progress_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")

    for epoch_number in progress_bar:
        # Train Mode
        model.train(True)

        avg_loss, avg_accuracy, avg_worst_group_accuracy = _train_one_epoch(
            epoch_number, writer, model, train_loader, loss_function, optimizer, device
        )

        writer.add_scalars(
            "Loss/Epoch",
            {
                "Training": avg_loss,
            },
            epoch_number,
        )

        writer.add_scalars(
            "Accuracy/Epoch",
            {
                "Training": avg_accuracy,
            },
            epoch_number,
        )

        writer.add_scalars(
            "Accuracy/Worst Group",
            {
                "Training": avg_worst_group_accuracy,
            },
            epoch_number,
        )

        evaluation_metrics = _evaluate_model(
            dataloaders=validation_loaders,
            model=model,
            writer=writer,
            device=device,
            loss_function=loss_function,
            epoch_number=epoch_number,
        )

        progress_bar.set_postfix(
            {
                "Train Loss": f"{avg_loss:.4f}",
                **{
                    f"Valid Loss {desc}": value
                    for desc, value in evaluation_metrics["Loss"].items()
                },
                "Train Accuracy": f"{avg_accuracy:.2%}",
                **{
                    f"Valid Accuracy {desc}": value
                    for desc, value in evaluation_metrics["Accuracy"].items()
                },
                **{
                    f"Valid Worst Group Accuracy {desc}": value
                    for desc, value in evaluation_metrics[
                        "Worst_group_accuracy"
                    ].items()
                },
            }
        )

        avg_vloss = evaluation_metrics["Loss"][list(validation_loaders.keys())[0]]

        # Save best model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            # Reset early stopping counter
            early_stopping_counter = 0

            model_directory_path = (
                f"models/{timestamp}_{experiment_description}"
                if experiment_description
                else f"models/{timestamp}"
            )

            os.makedirs(model_directory_path, exist_ok=True)

            model_path = os.path.join(model_directory_path, "model.pt")
            torch.save(model.state_dict(), model_path)

            with open(os.path.join(model_directory_path, "metadata.txt"), "w") as f:
                if experiment_description:
                    f.write(f"Description: {experiment_description}\n")

                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Epoch: {epoch_number} / {num_epochs} (Best Model)\n")

                for desc, value in evaluation_metrics["Loss"].items():
                    f.write(f"Validation Loss {desc}: {value:.4f}\n")

                f.write(f"Training Accuracy: {avg_accuracy:.4f}\n")

                for desc, value in evaluation_metrics["Accuracy"].items():
                    f.write(f"Validation Accuracy {desc}: {value:.4f}\n")

                f.write(
                    f"Training Worst Group Accuracy: {avg_worst_group_accuracy:.4f}\n"
                )

                for desc, value in evaluation_metrics["Worst_group_accuracy"].items():
                    f.write(f"Validation Worst Group Accuracy {desc}: {value:.4f}\n")

                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Optimizer: {optimizer_type.__name__}\n")
                f.write(f"Loss Function: {loss_function.__class__.__name__}\n")

        else:
            early_stopping_counter += 1

        if use_early_stopping and early_stopping_counter >= patience:
            print("Early Stopping triggered")
            break

    return model_path, tensorboard_log_dir_path


def deep_feature_reweighting(
    path_to_model: str,
    path_to_tensorboard_run: str,
    num_epochs: int,
    model: torch.nn.Module,
    validation_loaders: dict[str, DataLoader],
    train_loader: DataLoader,
    optimizer_type=torch.optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0,
    loss_function=torch.nn.CrossEntropyLoss(),
    use_early_stopping: bool = True,
    patience: int = 5,
) -> tuple[str, str]:
    """Apply DFR to an already pretrained model by retraining the last layer.

    Args:
        path_to_model (str): Path to the pre-trained model (state dict).
        path_to_tensorboard_run (str): Path to TensorBoard run.
        num_epochs (int): Number of epochs.
        model (torch.nn.Module): Model architecture.
        validation_loaders (dict[str, DataLoader]): A dictionary containing all Dataloader which should be used for evaluation. The key should serve as description.
        train_loader (DataLoader): DataLoader for train set.
        optimizer_type (_type_, optional): Optimizer used during training. Defaults to torch.optim.Adam.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.
        loss_function (_type_, optional): Loss function. Defaults to torch.nn.CrossEntropyLoss().
        use_early_stopping (bool, optional): Whether to use (True) or not use (False) early stopping. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement before triggering early stopping. Defaults to 5.

    Returns:
        tuple: (str, str) containing:
            - Path to the best retrained model.
            - Path to the TensorBoard log directory.
    """
    # Load pre-trained model state
    model.load_state_dict(torch.load(path_to_model, weights_only=True))

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last fully connected layer
    last_layer = list(model.children())[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    # Define the optimizer with only the last layers parameters
    optimizer = optimizer_type(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr,
        weight_decay=weight_decay,
    )

    writer = SummaryWriter(path_to_tensorboard_run)

    early_stopping_counter = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    best_vloss = math.inf
    model_path = None

    # Evaluate Model before retraining
    _evaluate_model(
        dataloaders=validation_loaders,
        model=model,
        writer=writer,
        epoch_number=0,
        device=device,
        loss_function=loss_function,
        reweighting=True,
    )

    progress_bar = tqdm(range(1, num_epochs + 1), desc="Reweighting Epochs")

    for epoch_number in progress_bar:
        model.train()

        # Train only the last layer
        avg_loss, avg_accuracy, avg_worst_group_accuracy = _train_one_epoch(
            epoch_number, writer, model, train_loader, loss_function, optimizer, device
        )

        writer.add_scalars(
            "DFR-Loss/Epoch",
            {
                "Training": avg_loss,
            },
            epoch_number,
        )

        writer.add_scalars(
            "DFR-Accuracy/Epoch",
            {
                "Training": avg_accuracy,
            },
            epoch_number,
        )

        writer.add_scalars(
            "DFR-Accuracy/Worst Group",
            {
                "Training": avg_worst_group_accuracy,
            },
            epoch_number,
        )

        evaluation_metrics = _evaluate_model(
            dataloaders=validation_loaders,
            model=model,
            writer=writer,
            epoch_number=epoch_number,
            device=device,
            loss_function=loss_function,
            reweighting=True,
        )

        progress_bar.set_postfix(
            {
                "Train Loss": f"{avg_loss:.4f}",
                **{
                    f"Valid Loss {desc}": value
                    for desc, value in evaluation_metrics["Loss"].items()
                },
                "Train Accuracy": f"{avg_accuracy:.2%}",
                **{
                    f"Valid Accuracy {desc}": value
                    for desc, value in evaluation_metrics["Accuracy"].items()
                },
                **{
                    f"Valid Worst Group Accuracy {desc}": value
                    for desc, value in evaluation_metrics[
                        "Worst_group_accuracy"
                    ].items()
                },
            }
        )

        avg_vloss = evaluation_metrics["Accuracy"][list(validation_loaders.keys())[0]]

        # Save best model based on validation loss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            # Reset early stopping counter
            early_stopping_counter = 0

            model_directory_path = os.path.dirname(path_to_model)

            model_path = os.path.join(model_directory_path, "dfr_model.pt")
            torch.save(model.state_dict(), model_path)

            with open(os.path.join(model_directory_path, "dfr_metadata.txt"), "w") as f:
                f.write("Retrained last layer only\n")
                f.write(f"Epoch: {epoch_number} / {num_epochs} (Best Model)\n")

                for desc, value in evaluation_metrics["Loss"].items():
                    f.write(f"Validation Loss {desc}: {value:.4f}\n")

                f.write(f"Training Accuracy: {avg_accuracy:.4f}\n")

                for desc, value in evaluation_metrics["Accuracy"].items():
                    f.write(f"Validation Accuracy {desc}: {value:.4f}\n")

                f.write(
                    f"Training Worst Group Accuracy: {avg_worst_group_accuracy:.4f}\n"
                )

                for desc, value in evaluation_metrics["Worst_group_accuracy"].items():
                    f.write(f"Validation Worst Group Accuracy {desc}: {value:.4f}\n")

                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Optimizer: {optimizer_type.__name__}\n")
                f.write(f"Loss Function: {loss_function.__class__.__name__}\n")

        else:
            early_stopping_counter += 1

        if use_early_stopping and early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    return model_path, path_to_tensorboard_run


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
    total_loss = 0.0
    num_predictions = defaultdict(int)
    num_correct_predictions = defaultdict(int)

    N = len(train_loader) // 2 - 1

    for i, data in enumerate(train_loader):
        inputs, true_labels, encoded_labels, spurious = data

        inputs, true_labels, encoded_labels, spurious = (
            inputs.to(device),
            true_labels.to(device),
            encoded_labels.to(device),
            spurious.to(device),
        )

        # Set gradients to zero
        optimizer.zero_grad()

        # Fetch predictions
        outputs = model(inputs)

        # Get predicted labels
        _, predicted_labels = torch.max(outputs, 1)

        for pred, encoded_label, true_label, spur in zip(
            predicted_labels, encoded_labels, true_labels, spurious
        ):
            key = f"{true_label} - {'spurious' if spur else 'not spurious'}"

            if pred.item() == encoded_label.item():
                num_correct_predictions[key] += 1

            num_predictions[key] += 1

        # Compute the loss
        loss = loss_function(outputs, encoded_labels)
        loss.backward()

        # Perform optimization step
        optimizer.step()

        # Add loss
        running_loss += loss.item()
        total_loss += loss.item()

        if i % N == N:
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

            tb_x = (current_epoch - 1) * len(train_loader) + i

            tb_writer.add_scalar("Loss/Batch Training Loss", last_loss, tb_x)
            tb_writer.add_scalar(
                "Accuracy/Overall Training Accuracy", overall_accuracy, tb_x
            )
            tb_writer.add_scalar(
                "Accuracy/Worst Group Training Accuracy", worst_group_accuracy, tb_x
            )

            for group, accuracy in group_accuracies.items():
                tb_writer.add_scalar(f"Accuracy/Training/{group}", accuracy, tb_x)

            running_loss = 0.0

    avg_loss = total_loss / len(train_loader)

    group_accuracies = {
        group: num_correct_predictions[group] / num_predictions[group]
        for group in num_predictions
    }

    worst_group_accuracy = min(group_accuracies.values())

    total_correct = sum(num_correct_predictions.values())
    total_predictions = sum(num_predictions.values())
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    return avg_loss, overall_accuracy, worst_group_accuracy


def _evaluate_model(
    dataloaders: dict[str, DataLoader],
    model: torch.nn.Module,
    writer: SummaryWriter,
    epoch_number: int,
    device: str,
    loss_function,
    reweighting=False,
):
    # Evaluation Mode
    model.eval()
    metrics = defaultdict(dict)
    all_group_accuracies = dict()

    for describtion, dataloader in dataloaders.items():
        running_vloss = 0.0
        num_predictions = defaultdict(int)
        num_correct_predictions = defaultdict(int)

        # Disable gradient computation
        with torch.no_grad():
            for i, vdata in enumerate(dataloader):
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
        all_group_accuracies[describtion] = group_vaccuracies

        total_correct = sum(num_correct_predictions.values())
        total_predictions = sum(num_predictions.values())
        avg_vaccuracy = (
            total_correct / total_predictions if total_predictions > 0 else 0
        )

        metrics["Loss"][describtion] = avg_vloss
        metrics["Accuracy"][describtion] = avg_vaccuracy
        metrics["Worst_group_accuracy"][describtion] = avg_worst_group_vaccuracy

    # Log metrics to TensorBoard
    writer.add_scalars(
        "DFR-Loss/Epoch" if reweighting else "Loss/Epoch",
        {f"Validation/{desc}": value for desc, value in metrics["Loss"].items()},
        epoch_number,
    )

    writer.add_scalars(
        "DFR-Accuracy/Epoch" if reweighting else "Accuracy/Epoch",
        {f"Validation/{desc}": value for desc, value in metrics["Accuracy"].items()},
        epoch_number,
    )

    writer.add_scalars(
        "DFR-Accuracy/Worst Group" if reweighting else "Accuracy/Worst Group",
        {
            f"Validation/{desc}": value
            for desc, value in metrics["Worst_group_accuracy"].items()
        },
        epoch_number,
    )

    for desc, group_accuracies in all_group_accuracies.items():
        for group, accuracy in group_accuracies.items():
            writer.add_scalar(
                f"DFR-Group Accuracy/Validation/{desc}/{group}"
                if reweighting
                else f"Group Accuracy/Validation/{desc}/{group}",
                accuracy,
                epoch_number,
            )

    writer.flush()

    return metrics
