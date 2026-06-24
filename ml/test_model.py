import argparse
import datetime
import gc
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from load_config import load_config
from ml.train_model import get_best_params, get_model
from ml.utils.conf_intervals import aggregate_metrics
from ml.utils.data import get_class_distribution, get_data_loaders, get_filenames, get_parameters
from ml.utils.training import get_device, get_evaluation_metrics, get_predictions

config = load_config()
RESULTS_DIR = config["paths"]["ml_results_dir"]
VALID_DATASET_NAMES = config["valid_dataset_names"]
VALID_MODELS = ["cnn", "cnn_bigru", "cnn_bilstm", "cnn_transf"]


def test_model(model_path, model_type, test_loader, filename, source_dataset):
    """
    Test a pre-trained model on a given test loader (direct transfer, no fine-tuning).

    Args:
        model_path (str or Path): Path to the saved model state dict (.pt file)
        model_type (str): Model type (e.g. "cnn")
        test_loader: DataLoader for the test set
        filename (str): Name of the dataset CSV file
        source_dataset (str): Name of dataset that model was trained on e.g. "iridia_af"

    Returns:
        tuple: (metrics dict, predictions dict)
    """
    device = get_device()
    print(f"Using device: {device}\n")
    print(f"Loading model from: {model_path}\n")

    dataset_id = filename.replace(".csv", "").replace("dataset_", "")
    train_params = get_parameters(filename, source_dataset)
    input_size = int(train_params["input_window_size"] * 60 * train_params["sampling_rate"])

    best_params = get_best_params(source_dataset, dataset_id)
    if best_params is None:
        print("Using default hyperparameters for CNN...")
        dropout_rate = 0.3

        if input_size <= 3000:
            adaptive_pool_size = 64
        elif input_size <= 12000:
            adaptive_pool_size = 128
        elif input_size < 40000:
            adaptive_pool_size = 512
        else:
            adaptive_pool_size = 1024

        base_out_channels = 32
        base_kernel_size = 5

    else:
        dropout_rate = best_params["dropout_rate"]
        adaptive_pool_size = best_params["adaptive_pool_size"]
        base_out_channels = best_params["base_out_channels"]
        base_kernel_size = best_params["base_kernel_size"]

    if model_type in ["cnn_bigru", "cnn_bilstm", "cnn_transf"]:
        if input_size <= 3000:        
            adaptive_pool_size = 64
        elif input_size <= 60000:     
            adaptive_pool_size = 128
        else:
            adaptive_pool_size = 512

    # load model
    model = get_model(model_type=model_type, device=device, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # use the source model's saved threshold
    if "threshold" not in checkpoint:
        raise ValueError(f"No threshold found in checkpoint: {model_path}")
    threshold = float(checkpoint["threshold"])
    print(f"Using source threshold: {threshold}")

    predictions = get_predictions(model, test_loader, device, threshold=threshold)
    metrics = get_evaluation_metrics(predictions)
    metrics["threshold"] = threshold

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics, predictions


def test_multiple_models(source_dataset, target_dataset, model_type, start_idx, end_idx, cv_mode="kfold"):
    """
    Test one or more pre-trained models on an external dataset.

    Args:
        source_dataset (str): Name of dataset that model was trained on e.g. "iridia_af"
        target_dataset (str): Name of dataset to test on e.g. "afdb"
        model_type (str): Model type e.g. "cnn"
        start_idx (int): Start index for filenames
        end_idx (int): End index for filenames (None = to end)
        cv_mode (str):
            - "kfold": Tests the single final model on each fold's test set and aggregates with 95% CI.
            - "single": Tests the single final model on a single train-validation-test split.
    """
    if (source_dataset not in VALID_DATASET_NAMES) or (target_dataset not in VALID_DATASET_NAMES):
        raise ValueError(f"Dataset names must be in: {VALID_DATASET_NAMES}")

    filenames = get_filenames(target_dataset)

    if end_idx is not None:
        selected_files = filenames[start_idx:end_idx]
    else:
        selected_files = filenames[start_idx:]

    if not selected_files:
        raise FileNotFoundError("No files selected with the given indices.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    if cv_mode == "kfold":
        eval_type = "cross-validation"
        iteration_name = "fold"
    else:
        eval_type = "single train-test split"
        iteration_name = "run"

    print(f"\n{'='*70}")
    print(f"Selected files: {selected_files}")
    print(f"Total files to process: {len(selected_files)}")
    print(f"Source dataset: {source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"Model: {model_type}")
    print(f"Type of evaluation: {eval_type}")
    print(f"{'='*70}\n")

    for idx, filename in enumerate(selected_files, 1):
        print(f"Dataset {idx} out of {len(selected_files)}: {filename}")
        dataset_id = filename.replace(".csv", "").replace("dataset_", "")  

        # match the source model on both prediction horizon and input window
        hor_inp_match = re.search(r"hor[\d.]+_inp[\d.]+", dataset_id)
        if hor_inp_match is None:
            print(f"✗ Could not parse horizon/input window from {dataset_id}, skipping.\n")
            continue
        hor_inp = hor_inp_match.group(0)

        # fetch final trained model
        model_path = list((RESULTS_DIR / source_dataset / model_type).glob(f"final/{hor_inp}_*/model.pt"))
        if not model_path:
            print(f"✗ No final model found for {dataset_id} (matched on {hor_inp}), skipping.\n")
            continue
        model_path = model_path[0]
        print(f"Selected final model: {model_path}")

        class_dist = get_class_distribution(filename=filename, dataset_name=target_dataset)

        all_metrics = []

        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = RESULTS_DIR / target_dataset / model_type / f"pretrained/{dataset_id}_{run_timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        for _, _, test_loader, iteration, patient_list in get_data_loaders(filename=filename, dataset_name=target_dataset, cv_mode=cv_mode, balance_train=False):
            print(f"\nTesting {iteration_name} {iteration}...")
            try:
                metrics, predictions = test_model(model_path=model_path, model_type=model_type, test_loader=test_loader, filename=filename, source_dataset=source_dataset)
                metrics[iteration_name] = iteration
                all_metrics.append(metrics)

                np.savez(Path(folder, f"predictions_{iteration_name}{iteration}.npz"), y_true=predictions["y_true"], y_pred_proba=predictions["y_pred_proba"], y_pred=predictions["y_pred"])
                print(f"Predictions saved to {folder / f'predictions_{iteration_name}{iteration}.npz'}")

            except Exception as e:
                traceback.print_exc()
                print(f"✗ Error on {iteration_name} {iteration}: {e}")
                continue

        if not all_metrics:
            print(f"✗ No results for {filename}, skipping.\n")
            continue

        print(f"\n{'='*70}")
        print(f"Test Results")
        print(f"{'='*70}")
        print(f"Class Distribution: {class_dist['formatted']}")

        if cv_mode == "kfold":
            final_metrics = aggregate_metrics(all_metrics)
            final_metrics["class_dist"] = class_dist["formatted"]
            print(f"Summary for all {iteration_name}s (95% CI)")
            for metric_name, display_name in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("accuracy", "Accuracy"), ("recall", "Recall"), ("precision", "Precision"), ("f1_score", "F1 Score"), ("specificity", "Specificity"), ("brier", "Brier"), ("bss", "BSS"), ("ece", "ECE")]:
                mean = final_metrics[f"{metric_name}_mean"]
                ci_lower = final_metrics[f"{metric_name}_ci_lower"]
                ci_upper = final_metrics[f"{metric_name}_ci_upper"]
                std = final_metrics[f"{metric_name}_std"]
                dp = 4 if metric_name in ("brier", "bss", "ece") else 2
                print(f"{display_name:15s}: {mean:.{dp}f} (95% CI: [{ci_lower:.{dp}f}, {ci_upper:.{dp}f}], SD: {std:.{dp}f})")
        else:
            final_metrics = all_metrics[0]
            final_metrics["class_dist"] = class_dist["formatted"]
            print(f"Summary results for single {iteration_name}")
            for metric_name, display_name in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("accuracy", "Accuracy"), ("recall", "Recall"), ("precision", "Precision"), ("f1_score", "F1 Score"), ("specificity", "Specificity"), ("brier", "Brier"), ("bss", "BSS"), ("ece", "ECE")]:
                dp = 4 if metric_name in ("brier", "bss", "ece") else 2
                print(f"{display_name:15s}: {final_metrics[metric_name]:.{dp}f}")

        print(f"{'='*70}\n")

        print(f"\n✓ Completed testing on dataset [{idx} out of {len(selected_files)}]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning with no fine-tuning")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for filenames")
    parser.add_argument("--end-idx", type=int, default=None, help="End index for filenames (None = to end)")
    parser.add_argument("--source-dataset", type=str, default="iridia_af", help="Dataset that model was trained on")
    parser.add_argument("--target-dataset", type=str, default="afdb", help="Dataset to test model on")
    parser.add_argument("--model-type", type=str, default="cnn", choices=VALID_MODELS, help="Model")
    parser.add_argument("--cv-mode", type=str, default="kfold", choices=["kfold", "single"], help="kfold: aggregate all fold models with 95% CI; single: single train-test split")

    args = parser.parse_args()
    test_multiple_models(args.source_dataset, args.target_dataset, args.model_type, args.start_idx, args.end_idx, args.cv_mode)