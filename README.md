# 🔍 Analyzing Spurious Correlations with Deep Feature Reweighting

This repository is dedicated to exploring the effect of **spurious correlations** in datasets and the application of **Deep Feature Reweighting** (DFR) to address them. The project uses the **MNIST handwritten digit dataset**, guided by the findings from the paper "[Deep Feature Reweighting for Spurious Correlation Elimination](https://openreview.net/forum?id=Zb6c8A-Fghk)".

## 🛠️ Project Setup

This project uses **Poetry** for dependency management.

### 🚀 Steps to Set Up

1. 📦 **Install Poetry** (if not already installed):
   ```bash
    pip install poetry
   ```
   For more installation options, refer to [Poetry’s documentation](https://python-poetry.org/docs/).

2. 📥 **Install dependencies:** Navigate to the project directory and install the required packages:
    ```bash
   poetry install
   ```

3. 🪄 **Activate the environment:**
    ```bash
   poetry shell
   ```

4. 🪝 **Set up pre-commit hooks:** This repository uses pre-commit hooks to ensure code quality. Install them by running:
    ```bash
   pre-commit install
   ```

## 🖥️ Running TensorBoard

This repository logs training runs in the `runs` directory. To monitor training progress and visualize results, use TensorBoard:
```bash
tensorboard --logdir=runs
```

## 📃 Repository Structure

### 📁 Folders:
- `mnist_data`: Stores the MNIST dataset, which will be automatically downloaded here on the first run.
- `models`: Stores trained models.
- `runs`: Directory where TensorBoard logs are stored for visualizing model training and evaluation.

### 📄 Files:
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `.pre-commit-config.yaml`: Configuration for pre-commit hooks to ensure code quality.
- `dataset.py`: Classes responsible for loading and preprocessing the MNIST dataset.
- `model_trainer.py`: Contains functions and classes to train the model, track performance, and handle spurious correlation analysis.
- `models.py`: Defines model architectures.
- `poetry.lock` and `pyproject.toml`: Poetry files for dependency management.
- `README.md`: Project documentation.
- `sandbox.ipynb`: A notebook for experimenting with spurious features in datasets, training models, and applying DFR.
- `spurious_features.py`: Functions for introducing spurious features into the MNIST images.
