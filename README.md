# Hybrid CNN Architectures for AF Episode Prediction

This repository contains accompanying code for our paper, "A Systematic Evaluation of Hybrid CNN Architectures for Atrial Fibrillation Episode Prediction Using ECG Data", to be presented at the [35th International Conference on Artificial Neural Networks (ICANN 2026)](https://e-nns.org/icann2026/), 14 to 17th September, in Padua, Italy.

## Install Dependencies

This project was built and tested using Python [3.12](https://www.python.org/downloads/release/python-3121/). The required packages can be installed in a virtual environment using Pip:
```bash
pip install -r requirements.txt
```

## Data

### Raw Data Download

Download the following datasets to the `data/raw` directory.

- [IRIDIA-AF](https://zenodo.org/records/8405941)
- [MIT-BIH Atrial Fibrillation Database (AFDB)](https://physionet.org/content/afdb/1.0.0/)

By default, the AFDB dataset will be downloaded to `physionet.org/files/afdb`. You will need to extract, rename, and reorganise the files to match the directory structure shown below.
```bash
gunzip -v *.gz
```

```
data/
└── raw/
    ├── afdb/
    │   ├── 00735.atr
    │   ├── 00735.hea
    │   ├── 00735.qrs
    │   ├── 03665.atr
    │   └── ...
    ├── iridia_af/
    │   ├── metadata.csv
    │   └── records/
    │       ├── record_000
    │       ├── record_001
    │       └── ...
```

### Data Preprocessing and Dataset Creation

To create the episode prediction datasets, run the following from the root folder:

```bash
python -m data.preprocessing.create_dataset
```

No output will be shown in the terminal. A directory `/ml/outputs/dataset_logs` will be automatically created and a timestamped log file will be saved inside it.

## Hyperparameter Optimisation

To run the hyperparameter optimisation script, run the following from the root folder:

```bash
python -m ml.utils.hyp_tuning
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset-name` | str | `iridia_af` | Dataset to use. Choices: `iridia_af`, `afdb` |
| `--model-type` | str | `cnn` | Model architecture to tune. Choices: `cnn` |
| `--file-indices` | int (one or more) | `None` (all files) | Specific file indices (0-based) to process. If omitted, all files are processed |
| `--resume` | flag | `False` | Resume from an existing Optuna study instead of starting fresh |
| `--cv` | flag | `False` | Use 5-fold cross-validation |

### Examples

Run with defaults:
```bash
python -m ml.utils.hyp_tuning
```

Tune on a specific dataset with cross-validation:
```bash
python -m ml.utils.hyp_tuning --dataset-name afdb --cv
```

Resume a previous study, processing only specific files:
```bash
python -m ml.utils.hyp_tuning --resume --file-indices 0 1 2
```

No output will be shown in the terminal. A directory `/ml/outputs/tuning_logs` will be automatically created and a timestamped log file will be saved inside it.

## Model Training & Evaluation

Three workflows are supported, covering training from scratch, evaluating direct transfer with no fine-tuning, and transfer learning with fine-tuning.

### 1. Training From Scratch

Trains a model from randomly initialized weights, using k-fold / leave-one-out / single-split cross-validation, or a `final` run with no held-out test set (for later transfer learning).

To run from the root folder:

```bash
python -m ml.train_model
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--start-idx` | int | `0` | Start index for filenames |
| `--end-idx` | int | `None` (to end) | End index for filenames |
| `--dataset-name` | str | `iridia_af` | Dataset name |
| `--model-type` | str | `cnn` | Model architecture. Choices: `cnn`, `cnn_bigru`, `cnn_bilstm`, `cnn_transf` |
| `--cv-mode` | str | `kfold` | Cross-validation mode. Choices: `kfold`, `loocv`, `single`, `final` |
| `--num-runs` | int | `1` | Number of runs on a single split for stability testing (only used with `--cv-mode single`) |
| `--folds` | int (one or more) | `None` (all folds) | Specific fold indices to run, e.g. `--folds 2 3` |
| `--balance-all` | flag | `False` | Balance train, val, and test sets to 50/50 |
| `--no-balance-train` | flag | `False` (training balancing is on by default) | Disable training set balancing |

**Cross-validation modes:**
- `kfold` — k-fold cross-validation
- `loocv` — leave-one-out cross-validation
- `single` — a single train/val/test split (no cross-validation)
- `final` — trains on the entire dataset with no held-out test set, for later use as a transfer source model

#### Examples

Train from scratch with default k-fold CV:
```bash
python -m ml.train_model
```

Train a specific architecture on a subset of files:
```bash
python -m ml.train_model --model-type cnn_bigru --start-idx 0 --end-idx 5
```

Train a final model (no test set) for later transfer learning:
```bash
python -m ml.train_model --cv-mode final
```

Re-run only specific folds:
```bash
python -m ml.train_model --cv-mode kfold --folds 0 1 2
```

### 2. Transfer Learning Without Fine-tuning

The testing script (`ml/test_model.py`) evaluates an already-trained **final** model directly on a different (target) dataset, with no fine-tuning. It uses the source model's saved decision threshold as-is.

To run from the root folder:

```bash
python -m ml.test_model
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--start-idx` | int | `0` | Start index for filenames |
| `--end-idx` | int | `None` (to end) | End index for filenames |
| `--source-dataset` | str | `iridia_af` | Dataset the model was trained on |
| `--target-dataset` | str | `afdb` | Dataset to test the model on |
| `--model-type` | str | `cnn` | Model architecture. Choices: `cnn`, `cnn_bigru`, `cnn_bilstm`, `cnn_transf` |
| `--cv-mode` | str | `kfold` | `kfold`: tests the single final model on each fold's test set and aggregates with 95% CI. `single`: tests on a single train/val/test split |

> **Prerequisite:** requires a `final` model already trained for `--source-dataset` (see Training From Scratch above) — the script looks for a matching `model.pt` under `RESULTS_DIR/<source_dataset>/<model_type>/final/`.

#### Examples

Test an `iridia_af`-trained final model directly on `afdb`:
```bash
python -m ml.test_model
```

Test on a subset of target files with a single split evaluation:
```bash
python -m ml.test_model --target-dataset afdb --cv-mode single --start-idx 0 --end-idx 3
```

### 3. Transfer Learning with Fine-tuning

Fine-tunes a pretrained `final` model on a new (target) dataset, using the same training script as from-scratch training. Fine-tuning runs in two stages: stage 1 trains only the model head (CNN frozen) for `--finetune-head-epochs` epochs, then stage 2 unfreezes the full network and trains at a reduced learning rate (`--finetune-lr-factor` × base LR).

To run from the root folder:

```bash
python -m ml.train_model --finetune
```

#### Options

In addition to the [from-scratch options](#options) above (`--dataset-name`, `--model-type`, `--cv-mode`, etc.), fine-tuning adds:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--finetune` | flag | `False` | Fine-tune from a pretrained model instead of training from scratch |
| `--finetune-head-epochs` | int | `5` | Epochs for stage 1 (head-only training) before unfreezing the full network |
| `--finetune-lr-factor` | float | `0.1` | LR multiplier for stage 2 (full network fine-tuning), e.g. `0.1` = 1/10th of base LR |

> **Prerequisites:**
> - Requires a `final` model already trained on `iridia_af` (see Training From Scratch above) — the script looks for a matching `model.pt` under `RESULTS_DIR/iridia_af/<model_type>/final/`.
> - `--finetune` cannot be combined with `--cv-mode final`.

#### Examples

Fine-tune a pretrained `iridia_af` model on `afdb` with default k-fold CV:
```bash
python -m ml.train_model --dataset-name afdb --finetune
```

Fine-tune with custom head-training length and LR factor:
```bash
python -m ml.train_model --dataset-name afdb --finetune --finetune-head-epochs 3 --finetune-lr-factor 0.05
```

## Citation

If you use this code or build on our work, please cite:

```bibtex
@inproceedings{nzomo_systematic_2026,
  title     = {A Systematic Evaluation of Hybrid CNN Architectures for Atrial Fibrillation Episode Prediction Using ECG Data},
  author    = {Mbithe Nzomo and Deshendran Moodley},
  booktitle = {Proceedings of the 35th International Conference on Artificial Neural Networks (ICANN 2026)},
  year      = {2026},
  address   = {Padua, Italy},
  note      = {To appear}
}
```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).