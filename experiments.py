"""This file contains a framework for running machine learning experiments with configurable datasets, models, and hyperparameters."""

from itertools import product

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import MNISTDataset
from model_trainer import deep_feature_reweighting, train
from models import SimpleModel
from spurious_features import SpuriousFeature


def run_experiments(
    experiment_title: str,
    labels: list[int] | None,
    main_spurious_features: dict[int, SpuriousFeature],
    minority_spurious_features: dict[int, SpuriousFeature],
    spurious_probabilities: dict[int, float],
    opposite_main_spurious_features: dict[int, SpuriousFeature],
    opposite_minority_spurious_features: dict[int, SpuriousFeature],
    opposite_spurious_probabilities: dict[int, float],
    dfr_main_spurious_features: dict[int, SpuriousFeature],
    dfr_minority_spurious_features: dict[int, SpuriousFeature],
    dfr_probabilities: dict[int, float],
    seed: int,
    experiment_config: dict[str, list[any]],
) -> list[str]:
    """Run multiple experiments where a model is trained on a spurious dataset and then retrained on an unbiased dataset with a range of given hyperparameters.

    Args:
        experiment_title (str): Title of experiment.
        labels (list[int] | None): Labels Labels that should be used in the dataset. Should be a list of integers (corresponding labels are then used) or None (all labels are used).
        spurious_features dict[int: SpuriousFeature]: Contains all spurious functions that should be applied to a given label (key in dictionary) in the train and a validation dataset.
        spurious_probabilities dict[int: float]: Contains the probabilities with which all specified spurious functions are applied to the corresponding label in the train and a validation dataset.
        opposite_spurious_features dict[int: SpuriousFeature]: Contains all spurious functions that should be applied to a given label (key in dictionary) in the opposite validation dataset.
        opposite_spurious_probabilities dict[int: float]: Contains the probabilities with which all specified spurious functions are applied to the corresponding label in the opposite validation dataset.
        seed: (int): Random seed.
        experiment_config (dict[str: list[any]]):
            A dictionary containing all hyperparameter values that should be tried in an experiment. The hyperparameter is the key and the value a list of all values.
            The following keys are allowed:
                - latent_size
                - learning_rate
                - weight_decay
                - batch_size
                - optimizer_type
                - num_epochs
                - num_dfr_epochs
                - model_architecture
                - use_early_stopping
                - patience

    Raises:
        ValueError: If there is an unexpected key in the experiment_config dictionary.

    Returns:
        list[str]: All tensorboard log paths to each individual experiment.
    """
    invalid_keys = set(experiment_config.keys()) - set(_base_config.keys())

    if invalid_keys:
        raise ValueError(f"The following keys are invalid: {invalid_keys}.")

    train_dataset = MNISTDataset(
        train=True,
        labels=labels,
        main_spurious_features=main_spurious_features,
        minority_spurious_features=minority_spurious_features,
        probabilities=spurious_probabilities,
        random_seed=seed,
    )

    validation_spurious_dataset = MNISTDataset(
        train=False,
        labels=labels,
        main_spurious_features=main_spurious_features,
        minority_spurious_features=minority_spurious_features,
        probabilities=spurious_probabilities,
        random_seed=seed,
    )

    validation_opposite_spurious_dataset = MNISTDataset(
        train=False,
        labels=labels,
        main_spurious_features=opposite_main_spurious_features,
        minority_spurious_features=opposite_minority_spurious_features,
        probabilities=opposite_spurious_probabilities,
        random_seed=seed,
    )

    validation_non_spurious_dataset = MNISTDataset(
        train=False, labels=labels, random_seed=seed
    )

    validation_only_spurious = MNISTDataset(
        train=False,
        labels=labels,
        main_spurious_features=main_spurious_features,
        minority_spurious_features=minority_spurious_features,
        probabilities={label: 1 for label in labels},
        random_seed=seed,
    )

    dfr_train_dataset = MNISTDataset(
        train=True,
        labels=labels,
        main_spurious_features=dfr_main_spurious_features,
        minority_spurious_features=dfr_minority_spurious_features,
        probabilities=dfr_probabilities,
        random_seed=seed,
    )

    tensorboard_logs = list()

    hyperparams = list(experiment_config.keys())
    hyperparams_values = list(experiment_config.values())

    all_combinations = list(product(*hyperparams_values))

    for hyperparam_combination in all_combinations:
        config = _base_config.copy()
        for hyperparam, val in zip(hyperparams, hyperparam_combination):
            config[hyperparam] = val

        desc = ", ".join(
            [
                f"{hyperparam}={val}"
                for hyperparam, val in zip(hyperparams, hyperparam_combination)
            ]
        )
        tensorboard_logs.append(
            run_single_experiment(
                train_dataset=train_dataset,
                spurious_eval_dataset=validation_spurious_dataset,
                non_spurious_eval_dataset=validation_non_spurious_dataset,
                opposite_spurious_dataset=validation_opposite_spurious_dataset,
                only_spurious_dataset=validation_only_spurious,
                dfr_train_dataset=dfr_train_dataset,
                hyperparam_description=desc,
                experiment_title=experiment_title,
                seed=seed,
                **config,
            )
        )

    return tensorboard_logs


def run_experiment_with_different_seeds(
    experiment_title: str,
    labels: list[int] | None,
    main_spurious_features: dict[int, SpuriousFeature],
    minority_spurious_features: dict[int, SpuriousFeature],
    spurious_probabilities: dict[int, float],
    opposite_main_spurious_features: dict[int, SpuriousFeature],
    opposite_minority_spurious_features: dict[int, SpuriousFeature],
    opposite_spurious_probabilities: dict[int, float],
    dfr_main_spurious_features: dict[int, SpuriousFeature],
    dfr_minority_spurious_features: dict[int, SpuriousFeature],
    dfr_probabilities: dict[int, float],
    seeds: list[int],
    fixed_config: dict[str, any],
) -> dict[int, str]:
    """
    Run multiple experiments with the same hyperparameters but different random seeds.

    Args:
        experiment_title (str): Title of experiment.
        labels (list[int] | None): Labels used in the dataset.
        main_spurious_features (dict[int, SpuriousFeature]): Spurious features for the main dataset.
        minority_spurious_features (dict[int, SpuriousFeature]): Spurious features for the minority dataset.
        spurious_probabilities (dict[int, float]): Probabilities of applying spurious features for main datasets.
        opposite_main_spurious_features (dict[int, SpuriousFeature]): Spurious features for opposite main datasets.
        opposite_minority_spurious_features (dict[int, SpuriousFeature]): Spurious features for opposite minority datasets.
        opposite_spurious_probabilities (dict[int, float]): Probabilities for opposite datasets.
        dfr_main_spurious_features (dict[int, SpuriousFeature]): DFR-specific main spurious features.
        dfr_minority_spurious_features (dict[int, SpuriousFeature]): DFR-specific minority spurious features.
        dfr_probabilities (dict[int, float]): DFR-specific probabilities.
        seeds (list[int]): List of random seeds for the experiments.
        fixed_config (dict[str, any]): A fixed set of hyperparameters for all experiments.

    Returns:
        dict[int, str]: A dictionary mapping each seed to its TensorBoard log path.
    """
    invalid_keys = set(fixed_config.keys()) - set(_base_config.keys())

    if invalid_keys:
        raise ValueError(f"The following keys are invalid: {invalid_keys}.")

    results = {}

    for seed in seeds:
        # Update datasets with the new seed
        train_dataset = MNISTDataset(
            train=True,
            labels=labels,
            main_spurious_features=main_spurious_features,
            minority_spurious_features=minority_spurious_features,
            probabilities=spurious_probabilities,
            random_seed=seed,
        )

        validation_spurious_dataset = MNISTDataset(
            train=False,
            labels=labels,
            main_spurious_features=main_spurious_features,
            minority_spurious_features=minority_spurious_features,
            probabilities=spurious_probabilities,
            random_seed=seed,
        )

        validation_opposite_spurious_dataset = MNISTDataset(
            train=False,
            labels=labels,
            main_spurious_features=opposite_main_spurious_features,
            minority_spurious_features=opposite_minority_spurious_features,
            probabilities=opposite_spurious_probabilities,
            random_seed=seed,
        )

        validation_non_spurious_dataset = MNISTDataset(
            train=False, labels=labels, random_seed=seed
        )

        validation_only_spurious = MNISTDataset(
            train=False,
            labels=labels,
            main_spurious_features=main_spurious_features,
            minority_spurious_features=minority_spurious_features,
            probabilities={label: 1 for label in labels},
            random_seed=seed,
        )

        dfr_train_dataset = MNISTDataset(
            train=True,
            labels=labels,
            main_spurious_features=dfr_main_spurious_features,
            minority_spurious_features=dfr_minority_spurious_features,
            probabilities=dfr_probabilities,
            random_seed=seed,
        )

        for hyperparam in fixed_config.keys():
            config = _base_config.copy()
            config[hyperparam] = fixed_config[hyperparam]

        desc = f"Seed={seed}, " + ", ".join(
            [f"{param}={value}" for param, value in fixed_config.items()]
        )

        tensorboard_log = run_single_experiment(
            train_dataset=train_dataset,
            spurious_eval_dataset=validation_spurious_dataset,
            non_spurious_eval_dataset=validation_non_spurious_dataset,
            opposite_spurious_dataset=validation_opposite_spurious_dataset,
            only_spurious_dataset=validation_only_spurious,
            dfr_train_dataset=dfr_train_dataset,
            hyperparam_description=desc,
            experiment_title=experiment_title,
            seed=seed,
            **config,
        )

        results[seed] = tensorboard_log

    return results


def run_single_experiment(
    train_dataset: Dataset,
    spurious_eval_dataset: Dataset,
    non_spurious_eval_dataset: Dataset,
    opposite_spurious_dataset: Dataset,
    only_spurious_dataset: Dataset,
    dfr_train_dataset: Dataset,
    batch_size: int,
    model_architecture: torch.nn.Module,
    latent_size: int,
    num_epochs: int,
    num_dfr_epochs: int,
    learning_rate: float,
    optimizer_type,
    weight_decay: float,
    use_early_stopping: bool,
    patience: int,
    hyperparam_description: str,
    seed: int,
    experiment_title: str,
) -> str:
    """Run a single experiment where a model is trained on a spurious dataset and then retrained on an unbiased dataset.

    Args:
        train_dataset (Dataset): Training dataset (includes spurious features).
        spurious_eval_dataset (Dataset): Evaluation dataset (includes same spurious features as train_dataset).
        non_spurious_eval_dataset (Dataset): Evaluation dataset (without any spurious features).
        opposite_spurious_dataset (Dataset): Evaluation dataset (includes "reversed" or "opposite" spurious features).
        only_spurious_eval_dataset (Dataset): Evaluation dataset (with only spurious features).
        dfr_train_dataset (Dataset): Unbiased train dataset for retraining the model.
        batch_size (int): Batch size.
        model_architecture (torch.nn.Module): Architecture or model class.
        latent_size (int): Size of the second last fully connected layer.
        num_epochs (int): Number of training epochs.
        num_dfr_epochs (int): Number of reweighting epochs.
        learning_rate (float): Learning rate.
        optimizer_type (_type_): Optimizer used for model training.
        weight_decay (float): Weight decay for the optimizer.
        use_early_stopping (bool): Whether to use (True) or not use (False) early stopping.
        patience (int): Number of epochs to wait for improvement before triggering early stopping.
        hyperparam_description (str): A short description of all hyperparameters that were manually set.
        seed (int): Random seed.
        experiment_title (str): Title of the experiment.

    Returns:
        str: Tensorboard log path.
    """
    print(f"Running {experiment_title} with: {hyperparam_description}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataloaders = {
        "validation_non_spurious": DataLoader(
            non_spurious_eval_dataset, batch_size=batch_size, shuffle=False
        ),
        "validation_spurious": DataLoader(
            spurious_eval_dataset, batch_size=batch_size, shuffle=False
        ),
        "validation_opposite_spurious": DataLoader(
            opposite_spurious_dataset, batch_size=batch_size, shuffle=False
        ),
        "validation_only_spurious": DataLoader(
            only_spurious_dataset, batch_size=batch_size, shuffle=False
        ),
    }

    model = model_architecture(num_classes=2, latent_size=latent_size)

    model_path, tensorboard_path = train(
        model=model,
        validation_loaders=validation_dataloaders,
        train_loader=train_loader,
        num_epochs=num_epochs,
        optimizer_type=optimizer_type,
        lr=learning_rate,
        experiment_description=f"{experiment_title}/{hyperparam_description}",
        weight_decay=weight_decay,
        use_early_stopping=use_early_stopping,
        patience=patience,
        seed=seed,
    )

    dfr_loader = DataLoader(dfr_train_dataset, batch_size=batch_size, shuffle=True)

    model_path, tensorboard_path = deep_feature_reweighting(
        path_to_model=model_path,
        path_to_tensorboard_run=tensorboard_path,
        model=model,
        num_epochs=num_dfr_epochs,
        validation_loaders=validation_dataloaders,
        train_loader=dfr_loader,
        optimizer_type=optimizer_type,
        lr=learning_rate,
        weight_decay=weight_decay,
        use_early_stopping=use_early_stopping,
        patience=patience,
        seed=seed,
    )

    return tensorboard_path


# Standard configuration of hyperparameters. Used if hyperparameter is not specified.
_base_config = {
    "latent_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 64,
    "optimizer_type": torch.optim.Adam,
    "num_epochs": 15,
    "num_dfr_epochs": 5,
    "model_architecture": SimpleModel,
    "use_early_stopping": True,
    "patience": 5,
}
