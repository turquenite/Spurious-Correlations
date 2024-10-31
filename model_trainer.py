import os
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(
    num_epochs,
    model: torch.nn.Module,
    validation_loader: DataLoader,
    train_loader: DataLoader,
    optimizer_type=torch.optim.Adam,
    lr=0.001,
    loss_function=torch.nn.CrossEntropyLoss(),
):
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

    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch_number, writer, model, train_loader, loss_function, optimizer, device
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
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

        avg_vloss = float(running_vloss / (i + 1))
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs("models", exist_ok=True)
            model_path = r"models\model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def train_one_epoch(
    epoch_index,
    tb_writer,
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function,
    optimizer,
    device,
):
    running_loss = 0.0
    last_loss = 0.0
    N = 100

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

        # Compute the loss and its gradients
        loss = loss_function(outputs, encoded_labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % N == N - 1:
            last_loss = running_loss / N
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss
