import argparse
import atexit
import datetime
import gc
import os
import re
import sys
import traceback
import uuid
from pathlib import Path

training_logs = "./ml/outputs/training_logs" 
os.makedirs(training_logs, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"{training_logs}/{timestamp}_{uuid.uuid4().hex[:8]}.log", "w")
atexit.register(log_file.close)
sys.stdout = log_file
sys.stderr = log_file

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


from load_config import load_config
from ml.models.cnn import CNNModel, CNNModelConfig
from ml.models.cnn_bigru import CNNBiGRUModel, CNNBiGRUModelConfig
from ml.models.cnn_bilstm import CNNBiLSTMModel, CNNBiLSTMModelConfig
from ml.models.cnn_transf import CNNTransformerModel, CNNTransformerModelConfig
from ml.utils.conf_intervals import aggregate_metrics
from ml.utils.data import get_class_distribution, get_data_loaders, get_filenames, get_parameters
from ml.utils.calibration import apply_calibrator, convert_to_binary, fit_calibrator
from ml.utils.training import configure_optimizers, estimate_loss, get_device, get_evaluation_metrics, get_median_threshold, get_optimal_threshold, get_constrained_threshold, get_predictions, get_warmup_lr, set_lr, load_pretrained_weights, set_cnn_frozen


config = load_config()
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEADS = config["data"]["leads"]
OUTPUTS_DIR = config["paths"]["ml_outputs_dir"]
PATIENCE = config["training"]["patience"]
RANDOM_SEED = config["random_seed"]
RESULTS_DIR = config["paths"]["ml_results_dir"]
VALID_DATASET_NAMES = config["valid_dataset_names"]
VALID_MODELS = config["valid_models"]


torch.manual_seed(RANDOM_SEED)

def get_best_params(dataset_name, dataset_id):
    """
    Load best CNN hyperparameters from JSON file.
    
    Args:
        dataset_name: Name of the dataset (e.g. "iridia_af")
        dataset_id: Dataset identifier (e.g. "hor0.5_inp5_ste1.5_tar0.5")
    Returns:
        best_params: Dictionary of best hyperparameters
    """
    best_params_path = OUTPUTS_DIR/"hyp_tuning"/dataset_name/"cnn"/f"best_params_{dataset_id}.json"

    try:
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        print(f"Loaded best parameters from {best_params_path}:")
        return best_params["hyperparameters"]
  
    except FileNotFoundError:   
        print(f"Best parameters file not found at {best_params_path}")
        return None

def get_batch_size(input_size_samples, base_out_channels, max_batch_size):
    """
    Choose a batch size that fits in memory for the given model config.

    Args:
        input_size_samples: Input length in samples 
        base_out_channels: First conv layer width
        max_batch_size: Ceiling from config (training.batch_size)
    Returns:
        batch_size
    """
    if input_size_samples >= 40000 and base_out_channels >= 64:
        return min(64, max_batch_size)
    elif input_size_samples >= 40000 or base_out_channels >= 64:
        return min(128, max_batch_size)
    else:
        return max_batch_size

def get_model(model_type, device, base_out_channels, base_kernel_size, adaptive_pool_size, dropout_rate, in_channels=len(LEADS)):
    """
    Factory function to create models.
    
    Args:
        model_type: Model type (e.g. "cnn")
        device: torch device
        base_out_channels: Out channels for first conv layer
        base_kernel_size: Kernel size for second conv layer
        adaptive_pool_size: Output size of AdaptiveAvgPool1d layer
        dropout_rate: Dropout rate to use in the model
        in_channels: Number of ECG channels 
    
    Returns:
        model: Initialized model on device
    """
    if model_type not in VALID_MODELS:
        raise ValueError(f"Model type '{model_type}' is not valid. Must be in: {VALID_MODELS}")

    if model_type == "cnn":
        model_config = CNNModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
        model = CNNModel(model_config)

    elif model_type == "cnn_bigru":
        model_config = CNNBiGRUModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
        model = CNNBiGRUModel(model_config)

    elif model_type == "cnn_bilstm":
        model_config = CNNBiLSTMModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
        model = CNNBiLSTMModel(model_config)

    elif model_type == "cnn_transf":
        model_config = CNNTransformerModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
        model = CNNTransformerModel(model_config)
  
    return model.to(device)

def train_model(filename, dataset_name, model_type, dataset_id, cv_mode="kfold", num_runs=1, balance_train=True, folds_to_run=None, balance_all=False, finetune=False, finetune_head_epochs=5, finetune_lr_factor=0.1, source_dataset="iridia_af"):
    """
    Args:
        filename (str): Name of the dataset CSV file
        dataset_name (str): Dataset name e.g. "iridia_af"
        model_type (str): Model type e.g. "cnn"
        dataset_id: Dataset identifier (e.g. "hor0.5_inp5_ste1.5_tar0.5")
        cv_mode (str):
            - "kfold": Uses k-fold cross-validation.
            - "loocv": Uses leave-one-out cross-validation.
            - "single": Uses a single train-validation-test split (no cross-validation).
            - "final": Trains on the entire dataset without a test set (for external validation).
        num_runs (int): Number of runs on single split for stability testing (only used when cv_mode="single").
            - If num_runs > 1: trains multiple times with different random seeds on the same data split.
        balance_train (bool): If True, use WeightedRandomSampler to balance training data
        folds_to_run (list, optional): List of fold indices to run (only used when cv=True). If None, runs all folds.
        balance_all (bool): If True, use WeightedRandomSampler to balance train, val, and test sets. Defaults to False.
        finetune (bool): If True, fine-tune from pretrained model instead of training from scratch. Defaults to False.
        finetune_head_epochs (int): Number of epochs to train head-only when finetuning from a pretrained model. Defaults to 5.
        finetune_lr_factor (float): Learning rate reduction factor for stage 2 of finetuning (full network training). Defaults to 0.1.
        source_dataset (str): Dataset the pretrained model was trained on (used when finetune is set). Defaults to "iridia_af".
    
    Returns:
        dict: Aggregated performance metrics across all iterations/folds.
    """   
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(f"Dataset '{dataset_name}' is not valid. Must be in: {VALID_DATASET_NAMES}")
    
    if cv_mode != "single" and num_runs > 1:
        raise ValueError("num_runs > 1 is only supported with single split mode")

    device = get_device()
    print(f"Using device: {device}\n")

    params = get_parameters(filename, dataset_name)
    input_size_samples = int(params["input_window_size"] * 60 * params["sampling_rate"])
    class_dist = get_class_distribution(filename=filename, dataset_name=dataset_name)

    metrics = []
    metrics_platt = []
    metrics_isotonic = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"hor{params['prediction_horizon']}_inp{params['input_window_size']}_tar{params['target_window_size']}_{timestamp}"

    if cv_mode == "final":
        iteration_name = "final"
    elif cv_mode == "loocv":
        iteration_name = "patient"
    elif cv_mode == "kfold":
        iteration_name = "fold"
    else:
        iteration_name = "run"

    if finetune:
        best_params = get_best_params(source_dataset, dataset_id)
    else:
        best_params = get_best_params(dataset_name, dataset_id)
    if best_params is None:
        print("Using default hyperparameters for CNN...")
        dropout_rate = 0.3
        learning_rate = 1e-4
        weight_decay = 1e-4

        if input_size_samples <= 3000:        # ~15s  
            adaptive_pool_size = 64
        elif input_size_samples <= 12000:     # ~1 min 
            adaptive_pool_size = 128
        elif input_size_samples < 40000:      # ~3 min 
            adaptive_pool_size = 512
        else:
            adaptive_pool_size = 1024

        base_out_channels = 32
        base_kernel_size = 5
        
    else:
        dropout_rate = best_params["dropout_rate"]
        adaptive_pool_size = best_params["adaptive_pool_size"]
        learning_rate = best_params["learning_rate"]
        weight_decay = best_params["weight_decay"]
        base_out_channels = best_params["base_out_channels"]
        base_kernel_size = best_params["base_kernel_size"]

    if model_type in ["cnn_bigru", "cnn_bilstm", "cnn_transf"]:
        if finetune:
            source_params = get_parameters(filename, source_dataset)
            source_input_size_samples = int(source_params["input_window_size"] * 60 * source_params["sampling_rate"])
            adaptive_pool_basis = source_input_size_samples
        else:
            adaptive_pool_basis = input_size_samples

        if adaptive_pool_basis <= 3000:
            adaptive_pool_size = 64
        elif adaptive_pool_basis <= 60000:
            adaptive_pool_size = 128
        else:
            adaptive_pool_size = 512
        
    if dataset_name == "iridia_af":
        max_norm = 5.0
        collapse_check = "once"
    else:
        # for smaller datasets with less training data, use more aggressive gradient clipping to mitgate against model collapse
        max_norm = 1.0
        collapse_check = "every" 

    batch_size = get_batch_size(input_size_samples, base_out_channels, BATCH_SIZE)

    if batch_size < BATCH_SIZE:
        scaled_lr = learning_rate * (batch_size / BATCH_SIZE)
        print(f"Batch size stepped down {BATCH_SIZE} -> {batch_size} for this config "
              f"(input_size_samples={input_size_samples}, base_out_channels={base_out_channels}); "
              f"scaling LR {learning_rate:.2e} -> {scaled_lr:.2e}")
        learning_rate = scaled_lr
    else:
        print(f"Batch size: {batch_size}")

    for train_loader, val_loader, test_loader, iteration, patient_list in get_data_loaders(filename=filename, dataset_name=dataset_name, cv_mode=cv_mode, num_runs=num_runs, balance_train=balance_train, balance_all=balance_all, batch_size=batch_size):

        if folds_to_run is not None and iteration is not None and iteration not in folds_to_run:
            print(f"Skipping {iteration_name} {iteration}")
            continue

        if cv_mode == "final":
            folder = RESULTS_DIR/dataset_name/model_type/"final"/model_id
        elif finetune:
            folder = RESULTS_DIR/dataset_name/model_type/"finetuned"/model_id/f"{iteration_name}{iteration}"
        else:
            folder = RESULTS_DIR/dataset_name/model_type/model_id/f"{iteration_name}{iteration}"
        folder.mkdir(parents=True, exist_ok=True)

        if num_runs > 1:
            run_seed = RANDOM_SEED + iteration
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            print(f"\n{'='*70}")
            print(f"Run {iteration}/{num_runs} (seed={run_seed})")
            print(f"{'='*70}\n")

        pretrained_found = True
        warmup_epochs = 2 if finetune else 3
        min_epochs_before_collapse_check = 3
        max_retries = 2 # number of retries for model collapse
        criterion = nn.BCEWithLogitsLoss()
        horizon = float(re.search(r"hor([\d.]+)", dataset_id).group(1))
        input_size_minutes = float(re.search(r"inp([\d.]+)", dataset_id).group(1))

        for retry in range(max_retries + 1):

            torch.manual_seed(RANDOM_SEED + (iteration or 0) + 1000 * retry)

            model = get_model(model_type=model_type, device=device, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)

            if finetune:
                model_path = list((RESULTS_DIR / source_dataset / model_type).glob(f"final/hor{horizon}_inp{input_size_minutes}_*/model.pt"))
                if not model_path:
                    pretrained_found = False
                    print(f"✗ No final model found for {dataset_id}, skipping.\n")
                    break
                model_path = model_path[0]

                pretrained_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if "pos_embedding" in pretrained_checkpoint["model_state_dict"]:
                    expected_pool_size = pretrained_checkpoint["model_state_dict"]["pos_embedding"].shape[1]
                    if expected_pool_size != adaptive_pool_size:
                        raise RuntimeError(
                            f"adaptive_pool_size mismatch for {dataset_id}: computed {adaptive_pool_size}, "
                            f"checkpoint's pos_embedding expects {expected_pool_size}."
                        )

                model = load_pretrained_weights(model, model_path, device)
                print(f"Fine-tuning model from: {model_path}")

                if "threshold" not in pretrained_checkpoint:
                    raise ValueError(f"No threshold found in pretrained checkpoint: {model_path}")
                prior_threshold = float(pretrained_checkpoint["threshold"])
                print(f"Prior threshold from pretrained model: {prior_threshold}")

                # stage 1: head only
                set_cnn_frozen(model, frozen=True)
                head_optimizer = configure_optimizers(model, learning_rate, weight_decay)
                print(f"Stage 1: head-only for {finetune_head_epochs} epochs")
                model.train()
                for epoch in range(finetune_head_epochs):
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        head_optimizer.zero_grad()
                        loss = criterion(model(x), y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        head_optimizer.step()
                    print(f"  Head epoch {epoch + 1}/{finetune_head_epochs} done")

                # stage 2: full network at reduced LR
                set_cnn_frozen(model, frozen=False)
                optimizer = configure_optimizers(model, learning_rate * finetune_lr_factor, weight_decay)
                print(f"Stage 2: full network, LR={learning_rate * finetune_lr_factor:.2e}")

            else:                
                optimizer = configure_optimizers(model, learning_rate, weight_decay)
            

            min_val_loss = float("inf")
            min_val_loss_epoch = 0
            best_model = None

            collapsed = False

            start_time = datetime.datetime.now()

            train_losses = []
            val_losses = []

            base_lr = learning_rate * finetune_lr_factor if finetune else learning_rate

            for epoch in range(EPOCHS):
                
                current_lr = get_warmup_lr(epoch, base_lr, warmup_epochs)
                set_lr(optimizer, current_lr)

                if epoch < warmup_epochs:
                    print(f"[Warmup] Epoch {epoch + 1}: LR = {current_lr:.2e}")

                # training
                model.train()
                batch_train_losses = []

                for batch_idx, (x, y) in enumerate(train_loader):
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    batch_train_losses.append(loss.item())

                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)
                    if batch_idx == 0:
                        print(f"Gradient norm: {total_norm:.2f}")
                    optimizer.step()

                train_loss = np.mean(batch_train_losses)
                train_losses.append(train_loss)

                if val_loader is None:
                    continue

                # validation
                val_loss = estimate_loss(model, device, val_loader, criterion)
                val_losses.append(val_loss)

                # predict with default 0.5 threshold
                val_predictions = get_predictions(model, val_loader, device)

                # check for model collapse and re-initialize
                if (collapse_check == "once" and epoch == warmup_epochs) or (collapse_check == "every" and epoch >= max(warmup_epochs, min_epochs_before_collapse_check)):
            
                    val_auroc = roc_auc_score(val_predictions["y_true"], val_predictions["y_pred_proba"])
                    
                    if val_auroc < 0.55:
                        collapsed = True
                        print(f"Detected potential model collapse (val_loss={val_loss:.2f}, AUROC={val_auroc:.2f})")
                        print("\nRetry number:", retry + 1)
                        print("\nContinuing to retry number:", retry + 2)
                        break 

                # threshold tuning on validation set
                if finetune:
                    threshold = get_constrained_threshold(val_predictions["y_true"], val_predictions["y_pred_proba"], prior_threshold)
                else:
                    threshold = get_optimal_threshold(val_predictions["y_true"], val_predictions["y_pred_proba"])
                print(f"Optimal threshold on val set: {threshold}")
                val_predictions["y_pred"] = (val_predictions["y_pred_proba"] >= threshold).astype(int)
                
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Train - Loss: {train_loss:.2f}")
                print(f"  Val   - Loss: {val_loss:.2f}\n")

                # early stopping
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_epoch = 0
                    best_model = model.state_dict()
                else:
                    min_val_loss_epoch += 1

                if min_val_loss_epoch >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if not collapsed:
                break

        if not pretrained_found:
            continue

        if collapsed:
            print(f"{iteration_name.capitalize()} {iteration} failed after {max_retries + 1} attempts.")

        if val_loader is None:
            folds_path = RESULTS_DIR/dataset_name/model_type
            # save final model
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_params": best_params,   
                "threshold": get_median_threshold(horizon=params["prediction_horizon"], input_window=params["input_window_size"], folds_path=folds_path),
                "train_losses": train_losses,
            }, Path(folder, "model.pt"))
            print(f"\nFinal model saved to {folder.absolute()}")
            continue

        # calibrators default to empty so a collapsed fold (best_model is None)
        # still saves a valid checkpoint without a NameError
        calibrators = {}
        calibrated_thresholds = {}

        if best_model is not None:
            print(f"Best validation loss: {min_val_loss:.2f}")
            model.load_state_dict(best_model)
            val_predictions = get_predictions(model, val_loader, device)
            if finetune:
                threshold = get_constrained_threshold(val_predictions["y_true"], val_predictions["y_pred_proba"], prior_threshold)
            else:
                threshold = get_optimal_threshold(val_predictions["y_true"], val_predictions["y_pred_proba"])

            # fit calibrators on the held-out validation split only, and recompute
            # the operating threshold on each calibrated scale (left unconstrained
            # even when finetuning, since calibration reshapes the probability
            # scale and prior_threshold was set on the uncalibrated scale)
            y_val = convert_to_binary(val_predictions["y_true"])
            for method in ("platt", "isotonic"):
                cal_val_proba, calibrators[method] = fit_calibrator(
                    val_predictions["y_pred_proba"], val_predictions["y_true"], method=method
                )
                calibrated_thresholds[method] = get_optimal_threshold(y_val, cal_val_proba)

        print(f"\nTesting {iteration_name} {iteration}...")
        test_loss = estimate_loss(model, device, test_loader, criterion)
        test_predictions = get_predictions(model, test_loader, device, threshold)

        end_time = datetime.datetime.now()
        time_taken = end_time - start_time  # fixed order
        hours, remainder = divmod(time_taken.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time taken for this {iteration_name}: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds\n")

        test_metrics = get_evaluation_metrics(test_predictions)
        test_metrics["loss"] = test_loss
        test_metrics[iteration_name] = iteration

        print(f"\nUncalibrated metrics (threshold={threshold:.4f}):")
        print(f"Loss: {test_loss:.2f}")
        print(f"Accuracy: {test_metrics['accuracy']:.2f}")
        print(f"Recall: {test_metrics['recall']:.2f}")
        print(f"Precision: {test_metrics['precision']:.2f}")
        print(f"F1 Score: {test_metrics['f1_score']:.2f}")
        print(f"Specificity: {test_metrics['specificity']:.2f}")
        print(f"AUROC: {test_metrics['auroc']:.2f}")
        print(f"AUPRC: {test_metrics['auprc']:.2f}")
        print(f"Brier: {test_metrics['brier']:.4f}")
        print(f"BSS: {test_metrics['bss']:.4f}")
        print(f"ECE: {test_metrics['ece']:.4f}\n")

        metrics.append(test_metrics)

        # calibrated test metrics 
        for method in ("platt", "isotonic"):
            if method not in calibrators:
                continue
            cal_test_proba = apply_calibrator(calibrators[method], test_predictions["y_pred_proba"])
            cal_thr = calibrated_thresholds[method]
            cal_pred = {
                "y_true": test_predictions["y_true"],
                "y_pred_proba": cal_test_proba,
                "y_pred": (cal_test_proba >= cal_thr).astype(int),
            }
            cal_metrics = get_evaluation_metrics(cal_pred)
            cal_metrics[iteration_name] = iteration

            print(f"\n[{method}] calibrated (threshold={cal_thr:.4f}):")
            print(f"Accuracy: {cal_metrics['accuracy']:.2f}")
            print(f"Recall: {cal_metrics['recall']:.2f}")
            print(f"Precision: {cal_metrics['precision']:.2f}")
            print(f"F1 Score: {cal_metrics['f1_score']:.2f}")
            print(f"Specificity: {cal_metrics['specificity']:.2f}")
            print(f"AUROC: {cal_metrics['auroc']:.2f}")
            print(f"AUPRC: {cal_metrics['auprc']:.2f}")
            print(f"Brier: {cal_metrics['brier']:.4f}")
            print(f"BSS: {cal_metrics['bss']:.4f}")
            print(f"ECE: {cal_metrics['ece']:.4f}\n")

            if method == "platt":
                metrics_platt.append(cal_metrics)
            else:
                metrics_isotonic.append(cal_metrics)

            pd.DataFrame(cal_metrics, index=[0]).to_csv(Path(folder, f"metrics_{method}.csv"))
            np.savez(
                Path(folder, f"predictions_{method}.npz"),
                y_true=cal_pred["y_true"],
                y_pred_proba=cal_pred["y_pred_proba"],
                y_pred=cal_pred["y_pred"],
            )
            print(f"Predictions saved to {folder / f'predictions_{method}.npz'}")

        # save model and other outputs for this iteration
        print(f"\nSaving model to {folder.absolute()}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_params": best_params,   
            "threshold": threshold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "calibrators": calibrators,
            "calibrated_thresholds": calibrated_thresholds,
        }, Path(folder, "model.pt"))
        pd.DataFrame(patient_list).to_csv(Path(folder, "patient_list.csv"))
        pd.DataFrame(params, index=[0]).to_csv(Path(folder, "parameters.csv"))
        pd.DataFrame(test_metrics, index=[0]).to_csv(Path(folder, "metrics_uncalibrated.csv"))

        # save predictions for visualisations
        np.savez(
            Path(folder, "predictions_uncalibrated.npz"),
            y_true=test_predictions["y_true"],
            y_pred_proba=test_predictions["y_pred_proba"],
            y_pred=test_predictions["y_pred"]
        )
        print(f"Predictions saved to {folder / 'predictions_uncalibrated.npz'}")

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    if cv_mode == "final":
        return {}
    
    print(f"\n{'='*70}")
    print(f"Test Results")
    print(f"{'='*70}")

    if cv_mode in ("kfold", "loocv"):
        try:
            final_metrics = aggregate_metrics(metrics)
            final_metrics["class_dist"] = class_dist["formatted"]

            for label, mlist in [
                ("Uncalibrated", metrics),
                ("Platt", metrics_platt),
                ("Isotonic", metrics_isotonic),
            ]:
                if not mlist:
                    continue
                agg = aggregate_metrics(mlist)
                print(f"\nSummary for all {iteration_name}s — {label} (95% CI)")
                for metric_name, display_name in [
                    ("auroc", "AUROC"),
                    ("auprc", "AUPRC"),
                    ("accuracy", "Accuracy"),
                    ("recall", "Recall"),
                    ("precision", "Precision"),
                    ("f1_score", "F1 Score"),
                    ("specificity", "Specificity"),
                    ("brier", "Brier"),
                    ("bss", "BSS"),
                    ("ece", "ECE"),
                ]:
                    mean = agg[f"{metric_name}_mean"]
                    ci_lower = agg[f"{metric_name}_ci_lower"]
                    ci_upper = agg[f"{metric_name}_ci_upper"]
                    std = agg[f"{metric_name}_std"]
                    dp = 4 if metric_name in ("brier", "bss", "ece") else 2
                    print(f"{display_name:15s}: {mean:.{dp}f} (95% CI: [{ci_lower:.{dp}f}, {ci_upper:.{dp}f}], SD: {std:.{dp}f})")

        except Exception as e:
            print(f"Too few folds to compute confidence intervals.")
            final_metrics = metrics[0]
            final_metrics["class_dist"] = class_dist["formatted"]
            
            # print results
            print(f"Summary results for single fold")
            for metric_name, display_name in [
                ("loss", "Test Loss"),
                ("auroc", "AUROC"),
                ("auprc", "AUPRC"),
                ("accuracy", "Accuracy"),
                ("recall", "Recall"),
                ("precision", "Precision"),
                ("f1_score", "F1 Score"),
                ("specificity", "Specificity"),
                ("brier", "Brier"),
                ("bss", "BSS"),
                ("ece", "ECE")
            ]:
                value = final_metrics[metric_name]
                dp = 4 if metric_name in ("brier", "bss", "ece") else 2
                print(f"{display_name:15s}: {value:.{dp}f}")

    elif num_runs > 1:
        
        final_metrics = {
            "class_dist": class_dist["formatted"],
            "num_runs": num_runs
        }
        
        print(f"Summary for {num_runs} runs on single split with different random seeds")
        for metric_name, display_name in [
            ("loss", "Test Loss"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("accuracy", "Accuracy"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("f1_score", "F1 Score"),
            ("specificity", "Specificity"),
            ("brier", "Brier"),
            ("bss", "BSS"),
            ("ece", "ECE")

        ]:
            values = [m[metric_name] for m in metrics if metric_name in m]
            mean = np.mean(values)
            std = np.std(values)
            min_ = np.min(values)
            max_ = np.max(values)
            
            final_metrics[f"{metric_name}_mean"] = mean
            final_metrics[f"{metric_name}_std"] = std
            final_metrics[f"{metric_name}_min"] = min_
            final_metrics[f"{metric_name}_max"] = max_
            
            dp = 4 if metric_name in ("brier", "bss", "ece") else 2
            print(f"{display_name:15s}: Mean - {mean:.{dp}f}; Std - {std:.{dp}f}; Min - {min_:.{dp}f}; Max - {max_:.{dp}f}")

    else:
        final_metrics = metrics[0]
        final_metrics["class_dist"] = class_dist["formatted"]
        
        # print results
        print(f"Summary results for single run on single split")
        for metric_name, display_name in [
            ("loss", "Test Loss"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("accuracy", "Accuracy"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("f1_score", "F1 Score"),
            ("specificity", "Specificity"),
            ("brier", "Brier"),
            ("bss", "BSS"),
            ("ece", "ECE")
        ]:
            value = final_metrics[metric_name]
            dp = 4 if metric_name in ("brier", "bss", "ece") else 2
            print(f"{display_name:15s}: {value:.{dp}f}")
    
    print(f"{'='*70}\n")

    return final_metrics

def train_multiple_models(dataset_name, model_type, start_idx, end_idx, cv_mode="kfold", num_runs=1, folds_to_run=None, balance_all=False, balance_train=True, finetune=False, finetune_head_epochs=5, finetune_lr_factor=0.1, source_dataset="iridia_af"):
    """
    Train model using one or more datasets using either k-fold cross-validation or a single train-validation-test split.
    
    Args:
        dataset_name (str): Dataset name e.g. "iridia_af"
        model_type (str): Model type e.g. "cnn"
        start_idx (int): Start index for filenames
        end_idx (int): End index for filenames (None = to end)
        cv_mode (str):
            - "kfold": Uses k-fold cross-validation.
            - "loocv": Uses leave-one-out cross-validation.
            - "single": Uses a single train-validation-test split (no cross-validation).
            - "final": Trains on the entire dataset without a test set (for external validation).
        num_runs (int): Number of runs on single split for stability testing.
        folds_to_run (list, optional): List of fold indices to run (only used when cv=True). If None, runs all folds.
        balance_all (bool): If True, use WeightedRandomSampler to balance train, val, and test sets. Defaults to False.
        balance_train (bool): If True, use WeightedRandomSampler to balance training data. Defaults to True.
        finetune (bool): If True, fine-tune from pretrained model instead of training from scratch. Defaults to False.
        finetune_head_epochs (int): Number of epochs to train head-only when finetuning from a pretrained model. Defaults to 5.
        finetune_lr_factor (float): Learning rate reduction factor for stage 2 of finetuning (full network training). Defaults to 0.1.
        source_dataset (str): Dataset the pretrained model was trained on (used when finetune is set). Defaults to "iridia_af".
    """

    filenames = get_filenames(dataset_name)

    if end_idx is not None:
        selected_files = filenames[start_idx:end_idx]
    else:
        selected_files = filenames[start_idx:]

    if not selected_files:
        raise FileNotFoundError("No files selected with the given indices.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*70}")
    print(f"Selected files: {selected_files}")
    print(f"Total files to process: {len(selected_files)}")
    print(f"Model: {model_type}")
    print(f"Training type: {'Fine-tuning' if finetune else 'From scratch'}")
    if finetune:
        print(f"Source dataset: {source_dataset}")
        print(f"Target dataset: {dataset_name}")
    else:
        print(f"Dataset: {dataset_name}")
    print(f"CV mode: {cv_mode}")
    print(f"{'='*70}\n")
    
    for idx, filename in enumerate(selected_files, 1):
        print(f"Dataset {idx} out of {len(selected_files)}: {filename}")
        dataset_id = filename.replace(".csv", "").replace("dataset_", "")
        
        try:
            train_model(filename=filename, dataset_name=dataset_name, model_type=model_type, dataset_id=dataset_id, cv_mode=cv_mode, num_runs=num_runs, folds_to_run=folds_to_run, balance_all=balance_all, balance_train=balance_train, finetune=finetune, finetune_head_epochs=finetune_head_epochs, finetune_lr_factor=finetune_lr_factor, source_dataset=source_dataset)
            print(f"\n✓ Completed training on dataset [{idx} out of {len(selected_files)}]\n")
            
        except Exception as e:
            traceback.print_exc()
            return(f"\n✗ Error on {filename}: {e}\n")

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run model on ECG datasets")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for filenames")
    parser.add_argument("--end-idx", type=int, default=None, help="End index for filenames (None = to end)")
    parser.add_argument("--dataset-name", type=str, default="iridia_af", choices=VALID_DATASET_NAMES, help="Dataset name")
    parser.add_argument("--model-type", type=str, default="cnn", choices=VALID_MODELS, help="Model")
    parser.add_argument("--cv-mode", type=str, default="kfold", choices=["kfold", "loocv", "single", "final"], help="Cross-validation mode - kfold, loocv, single, or final")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs for stability testing (use with --no-cv)")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Specific folds to run (e.g., --folds 2 3)")
    parser.add_argument("--balance-all", action="store_true", help="Balance train, val, and test sets to 50/50")
    parser.add_argument("--no-balance-train", dest="balance_train", action="store_false", help="Disable training set balancing (enabled by default)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune from pretrained model")
    parser.add_argument("--finetune-head-epochs", type=int, default=5, help="Number of epochs for stage 1 head-only training before unfreezing full network")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1, help="LR multiplier for stage 2 full network fine-tuning (e.g. 0.1 means 1/10th of base LR)")
    parser.add_argument("--source-dataset", type=str, default="iridia_af", choices=VALID_DATASET_NAMES, help="Dataset the pretrained model was trained on (used when --finetune is set)")
    
    args = parser.parse_args()

    if args.finetune and args.cv_mode == "final":
        parser.error("--finetune and --cv-mode final cannot be used together")

    train_multiple_models(args.dataset_name, args.model_type, args.start_idx, args.end_idx, args.cv_mode, args.num_runs, args.folds, args.balance_all, args.balance_train, finetune=args.finetune, finetune_head_epochs=args.finetune_head_epochs, finetune_lr_factor=args.finetune_lr_factor, source_dataset=args.source_dataset)