import argparse
import atexit
import gc
import json
import os
import pickle
import sys
from datetime import datetime

tuning_logs = "./ml/outputs/tuning_logs" 
os.makedirs(tuning_logs, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"{tuning_logs}/{timestamp}.log", "w")
atexit.register(log_file.close)
sys.stdout = log_file
sys.stderr = log_file

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.trial import Trial
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from load_config import load_config
from ml.models.cnn import CNNModel, CNNModelConfig
from ml.utils.data import get_data_loaders, get_parameters
from ml.utils.training import get_device

config = load_config()
EPOCHS = 25
PATIENCE = config["training"]["patience"]
RANDOM_SEED = config["random_seed"]
OUTPUTS_DIR = config["paths"]["ml_outputs_dir"]
DATASETS_DIR = config["paths"]["proc_data_dir"]
VALID_DATASET_NAMES = config["valid_dataset_names"]
VALID_MODELS = ["cnn", "cnn_bigru", "cnn_bilstm", "cnn_transf"]

torch.manual_seed(RANDOM_SEED)

def objective(trial, filename, dataset_name, model_type, dataset_id, results_dir, cv=False):
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
        filename: Dataset filename
        dataset_name: Dataset name
        model_type: Model to use (e.g. "cnn")
    
    Returns:
        brier score on validation set
    """
    device = get_device()
    params = get_parameters(filename, dataset_name)
    input_size = int(params["input_window_size"] * 60 * params["sampling_rate"])
    in_channels = len(config["data"]["leads"])
    

    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)


    if input_size <= 3000: # about 10s
        adaptive_pool_size = trial.suggest_categorical("adaptive_pool_size", [32, 64])
    elif input_size <= 12000: # about 60s
        adaptive_pool_size = trial.suggest_categorical("adaptive_pool_size", [64, 128, 256])
    elif input_size < 40000: # about 3 min
        adaptive_pool_size = trial.suggest_categorical("adaptive_pool_size", [256, 512, 1024])
    else: # about 5 min
        adaptive_pool_size = trial.suggest_categorical("adaptive_pool_size", [512, 1024, 2048])
    
    base_out_channels = trial.suggest_categorical("base_out_channels", [16, 32, 64])
    base_kernel_size = trial.suggest_categorical("base_kernel_size", [5, 7])

    cv_mode = "kfold" if cv else "single"

    if cv:
        fold_metrics = {"brier": [], "f1": [], "recall": [], "precision": []}
        data_iter = get_data_loaders(filename=filename, dataset_name=dataset_name, cv_mode=cv_mode)

        for fold_idx, (train_loader, val_loader, test_loader, iteration, patient_list) in enumerate(data_iter):
            print(f"  Trial {trial.number}, Fold {fold_idx + 1}/5")

            model_config = CNNModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
            
            model = CNNModel(model_config).to(device)
            
            optimiser = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )

            criterion = nn.BCEWithLogitsLoss()
            
            best_fold_brier = float("inf")
            best_fold_metrics = {}
            patience_counter = 0
            
            try:
                for epoch in range(EPOCHS):

                    model.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimiser.zero_grad()
                        logits = model(x)
                        loss = criterion(logits, y)
                        loss.backward()
                        optimiser.step()
                    
                    model.eval()
                    y_true, y_pred, y_pred_proba = [], [], []
                    
                    with torch.no_grad():
                        for x, y in val_loader:
                            x = x.to(device)
                            logits = model(x)
                            probabilities = torch.sigmoid(logits)
                            predictions = (probabilities > 0.5).float()
                            
                            y_true.extend(y.cpu().numpy())
                            y_pred.extend(predictions.cpu().numpy())
                            y_pred_proba.extend(probabilities.cpu().numpy())
                    
                    y_true = np.array(y_true).flatten()
                    y_pred = np.array(y_pred).flatten()
                    y_pred_proba = np.array(y_pred_proba).flatten()
                    
                    val_brier = brier_score_loss(y_true, y_pred_proba)
                    
                    if val_brier < best_fold_brier:
                        best_fold_brier = val_brier
                        best_fold_metrics = {
                            "brier": val_brier,
                            "f1": f1_score(y_true, y_pred, zero_division=0),
                            "recall": recall_score(y_true, y_pred, zero_division=0),
                            "precision": precision_score(y_true, y_pred, zero_division=0)
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= PATIENCE:
                        break
                
                for metric, value in best_fold_metrics.items():
                    fold_metrics[metric].append(value)
                    
            finally:
                del model, optimiser, train_loader, val_loader, test_loader
                gc.collect()
                torch.cuda.empty_cache()
            
            # early pruning based on running mean Brier
            if fold_idx >= 1:  
                running_mean_brier = np.mean(fold_metrics["brier"])
                trial.report(running_mean_brier, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        mean_brier = np.mean(fold_metrics["brier"])
        std_brier = np.std(fold_metrics["brier"])
        
        trial.set_user_attr("brier", mean_brier)
        trial.set_user_attr("brier_std", std_brier)
        trial.set_user_attr("f1", np.mean(fold_metrics["f1"]))
        trial.set_user_attr("f1_std", np.std(fold_metrics["f1"]))
        trial.set_user_attr("recall", np.mean(fold_metrics["recall"]))
        trial.set_user_attr("recall_std", np.std(fold_metrics["recall"]))
        trial.set_user_attr("precision", np.mean(fold_metrics["precision"]))
        trial.set_user_attr("precision_std", np.std(fold_metrics["precision"]))
        trial.set_user_attr("constraint_status", "valid")
        
        print(f"  Trial {trial.number} complete: Brier = {mean_brier:.4f} ± {std_brier:.4f}")
        
        return mean_brier
    
    data_iter = get_data_loaders(filename=filename, dataset_name=dataset_name, cv_mode=cv_mode)
    
    model_config = CNNModelConfig(in_channels=in_channels, base_out_channels=base_out_channels, base_kernel_size=base_kernel_size, adaptive_pool_size=adaptive_pool_size, dropout_rate=dropout_rate)
    
    model = CNNModel(model_config).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader, test_loader, iteration, patient_list = next(data_iter)
    
    best_brier = float("inf")
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    patience_counter = 0
    pruned = False
    
    try:
        for epoch in range(EPOCHS):

            # training
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimiser.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimiser.step()
            
            # validation
            model.eval()
            y_true = []
            y_pred = []
            y_pred_proba = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    logits = model(x)
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).float()

                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())
                    y_pred_proba.extend(probabilities.cpu().numpy())
            
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            y_pred_proba = np.array(y_pred_proba).flatten()

            val_brier = brier_score_loss(y_true, y_pred_proba) 
            val_f1 = f1_score(y_true, y_pred, zero_division=0)
            val_recall = recall_score(y_true, y_pred, zero_division=0)
            val_precision = precision_score(y_true, y_pred, zero_division=0)
            
            trial.report(val_brier, epoch)
            
            if trial.should_prune():
                pruned = True
                break
            
            if val_brier < best_brier:
                best_brier = val_brier
                best_f1 = val_f1
                best_recall = val_recall
                best_precision = val_precision
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                break
    
    finally:
        del train_loader
        del val_loader
        del test_loader
        del data_iter
        del model
        del optimiser
        
        gc.collect()
        torch.cuda.empty_cache()
    
    trial.set_user_attr("brier", best_brier)
    trial.set_user_attr("f1", best_f1)
    trial.set_user_attr("recall", best_recall)
    trial.set_user_attr("precision", best_precision)

    if pruned:
        trial.set_user_attr("constraint_status", "pruned")
        raise optuna.TrialPruned()
    
    trial.set_user_attr("constraint_status", "valid")
    return best_brier

class SaveBestTrialCallback:
    """Save best trial incrementally during optimisation."""
    
    def __init__(self, results_dir, dataset_id, study_name, initial_best_brier=float("inf"), cv=False):
        self.results_dir = results_dir
        self.dataset_id = dataset_id
        self.study_name = study_name
        self.best_brier = initial_best_brier
        self.cv = cv
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if trial.user_attrs.get("constraint_status") == "valid":
            current_brier = trial.user_attrs.get("brier", float("inf"))
            
            if current_brier < self.best_brier:
                self.best_brier = current_brier
                
                # save best params immediately
                best_params_path = self.results_dir / f"best_params_{self.dataset_id}.json"
                if self.cv:
                    performance = {
                        "brier": trial.user_attrs["brier"],
                        "brier_std": trial.user_attrs["brier_std"],
                        "f1": trial.user_attrs["f1"],
                        "f1_std": trial.user_attrs["f1_std"],
                        "recall": trial.user_attrs["recall"],
                        "recall_std": trial.user_attrs["recall_std"],
                        "precision": trial.user_attrs["precision"],
                        "precision_std": trial.user_attrs["precision_std"]
                    }
                else:
                    performance = {
                        "brier": trial.user_attrs["brier"],
                        "f1": trial.user_attrs["f1"],
                        "recall": trial.user_attrs["recall"],
                        "precision": trial.user_attrs["precision"]
                    }
                save_data = {
                    "hyperparameters": trial.params,
                    "performance": performance,
                    "metadata": {
                        "study_name": self.study_name,
                        "cv": self.cv,
                        "trial_number": trial.number,
                        "trial_value": trial.value,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                with open(best_params_path, "w") as f:
                    json.dump(save_data, f, indent=2)
                
                print(f"\n{'='*70}")
                print(f"NEW BEST SAVED! Trial {trial.number}: brier={current_brier:.4f}")
                print(f"Saved to: {best_params_path}")
                print(f"{'='*70}\n")
        
        # save complete study after each trial
        study_path = self.results_dir / f"{self.study_name}_study.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)

def optimise_hyperparameters(filename, dataset_name, model_type, dataset_id, results_dir, n_trials=100, timeout=None, resume=False, cv=False):
    """
    Run hyperparameter optimisation for specific dataset configuration.
    
    Args:
        filename: Dataset filename (e.g. "hor5_inp3.csv")
        dataset_name: Dataset name (e.g. "iridia_af")
        model_type: Model to use (e.g. "cnn")
        results_dir: Directory to save results
        n_trials: Number of trials (default: 100)
        timeout: Time limit in seconds (default: None)
        resume: Whether to resume from existing study (default: False)
    
    Returns:
        Optuna study object
    """
    completed_trials = 0
    remaining_trials = n_trials
    study = None

    if resume:
        study_files = list(results_dir.glob(f"{dataset_name}_{model_type}_{dataset_id}_*_study.pkl"))
        if study_files:
            latest_study_file = max(study_files, key=lambda p: p.stat().st_mtime)
            print(f"Found existing study: {latest_study_file}")
            with open(latest_study_file, "rb") as f:
                study = pickle.load(f)
            study_name = study.study_name
            completed_trials = len(study.trials)
            remaining_trials = n_trials - completed_trials
            print(f"Resuming from {len(study.trials)} completed trials")
            print(f"Remaining trials to run: {remaining_trials}")
            
        else:
            print("No existing study found to resume. Starting new study.")

    if study is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{dataset_name}_{model_type}_{dataset_id}_{timestamp}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=10
            ),
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )

    # load previous best brier if resuming
    initial_best_brier = float("inf")
    
    if resume:
        best_params_path = results_dir / f"best_params_{dataset_id}.json"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                existing_best = json.load(f)
            if cv:
                initial_best_brier = existing_best["performance"]["brier"]
            else:
                initial_best_brier = existing_best["performance"]["brier"]
            print(f"Loaded previous best Brier: {initial_best_brier:.4f}")

    save_callback = SaveBestTrialCallback(
        results_dir=results_dir,
        dataset_id=dataset_id,
        study_name=study_name,
        initial_best_brier=initial_best_brier,
        cv=cv
    )
    
    print(f"\n{'='*70}")
    print(f"Optuna hyperparameter optimisation")
    print(f"{'='*70}")
    print(f"Study: {study_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset configs: {dataset_id}")
    print(f"Trials: {n_trials}")
    print(f"Objective: Minimise Brier score")
    print(f"{'='*70}\n")

    study.optimize(
        lambda trial: objective(trial, filename, dataset_name, model_type, dataset_id, results_dir, cv),
        n_trials=remaining_trials,
        timeout=timeout,
        callbacks=[save_callback],
        show_progress_bar=True
    )
    
    print(f"\n{'='*70}")
    print(f"Optimisation Complete - {dataset_id}")
    print(f"{'='*70}")
    
    valid_trials = [t for t in study.trials if t.value is not None]
    
    if not valid_trials:
        print("No valid trials found.")
        return study

    valid_trials.sort(key=lambda t: t.value) 
    best = valid_trials[0]
    
    print(f"\nFound {len(valid_trials)} valid solutions")
   
    if cv:
        print(f"\nBest Solution (across 5 folds):")
        print(f"  Brier: {best.user_attrs['brier']:.4f} ± {best.user_attrs['brier_std']:.4f}")
        print(f"  F1:    {best.user_attrs['f1']:.4f} ± {best.user_attrs['f1_std']:.4f}")
        print(f"  Recall: {best.user_attrs['recall']:.4f}")
        print(f"  Precision: {best.user_attrs['precision']:.4f}")
    else:
        print(f"\nBest Solution for single split:")
        print(f"  Brier: {best.user_attrs['brier']:.4f}")
        print(f"  F1: {best.user_attrs['f1']:.4f}")
        print(f"  Recall: {best.user_attrs['recall']:.4f}")
        print(f"  Precision: {best.user_attrs['precision']:.4f}")
    
    print(f"\n  Hyperparameters:")
    for key, value in best.params.items():
        if key == "learning_rate" or key == "weight_decay":
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    print(f"\n{'='*70}")
    print(f"Top 5 Solutions:")
    print(f"{'='*70}")
    for i, t in enumerate(valid_trials[:5], 1):
        if cv:
            brier = t.user_attrs.get("brier", 0)
            f1 = t.user_attrs.get("f1", 0)
            precision = t.user_attrs.get("precision", 0)
            recall = t.user_attrs.get("recall", 0)
            std = t.user_attrs.get("brier_std", 0)
            print(f"\n{i}. Trial {t.number}: brier: {brier:.4f}±{std:.4f}, F1: {f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
        else:
            brier = t.user_attrs.get("brier", 0)
            f1 = t.user_attrs.get("f1", 0)
            precision = t.user_attrs.get("precision", 0)
            recall = t.user_attrs.get("recall", 0)
            print(f"\n{i}. Trial {t.number}: brier: {brier:.4f}, F1: {f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

    if valid_trials:
        # save best params
        best_params_path = results_dir/f"best_params_{dataset_id}.json"
        if cv:
            performance = {
                "brier": best.user_attrs["brier"],
                "brier_std": best.user_attrs["brier_std"],
                "f1": best.user_attrs["f1"],
                "f1_std": best.user_attrs["f1_std"],
                "recall": best.user_attrs["recall"],
                "recall_std": best.user_attrs["recall_std"],
                "precision": best.user_attrs["precision"],
                "precision_std": best.user_attrs["precision_std"]
            }
        else:
            performance = {
                "brier": best.user_attrs["brier"],
                "f1": best.user_attrs["f1"],
                "recall": best.user_attrs["recall"],
                "precision": best.user_attrs["precision"]
            }
        save_data = {
            "hyperparameters": best.params,
            "performance": performance,
            "metadata": {
                "study_name": study_name,
                "dataset": filename,
                "value": best.value,
                "n_trials": len(study.trials),
                "valid_trials": len(valid_trials)
            }
        }
        with open(best_params_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nBest parameters saved: {best_params_path}")

    study_path = results_dir/f"{study_name}_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Study saved: {study_path}")
    
    return study
    
def run_optimisation_function(dataset_name, model_type, file_indices=None, resume=False, cv=False):
    """
    Run hyperparameter optimisation for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        model_type: Model to use (e.g. "cnn")
        file_indices: Optional indices of specific files to process (0-based).
                   If None, processes all files.
        resume: Whether to resume from existing study (default: False)
        cv: Whether to use cross-validation (default: False)
    """
    results_dir = OUTPUTS_DIR/"hyp_tuning"/dataset_name/model_type
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = DATASETS_DIR/dataset_name
    filenames = sorted(list(dataset_dir.glob('*.csv')), key=lambda fp: get_parameters(fp.name, dataset_name)["prediction_horizon"])

    print(f"\nFound {len(filenames)} datasets in {dataset_dir}:\n")

    if file_indices is not None:
        filenames_to_process = []
        for idx in file_indices:
            if idx < 0 or idx >= len(filenames):
                print(f"Error: File index {idx} out of range [0, {len(filenames)-1}]")
                sys.exit(1)
            filenames_to_process.append(filenames[idx])  # was: [filenames[file_indices]]
        print(f"Processing {len(filenames_to_process)} file(s): {filenames_to_process}\n")  # was: filenames_to_process[0].name
    else:
        filenames_to_process = filenames
        print(f"Processing all {len(filenames)} files\n")

    for path in filenames_to_process:
        filename = path.name
        dataset_id = filename.replace("dataset_", "").replace(".csv", "")
        print("\n" + "="*70)
        print(f"Starting hyperparameter tuning for dataset: {dataset_name}, model_type: {model_type}, dataset configs: {dataset_id}, cv: {cv}")

        optimise_hyperparameters(filename=filename, dataset_name=dataset_name, model_type=model_type, dataset_id=dataset_id, results_dir=results_dir, resume=resume, cv=cv)
        
        print("Optimisations complete!")
        print("\n" + "="*70)
        print(f"\nResults saved in: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters")
    parser.add_argument("--dataset-name", type=str, default="iridia_af", choices=VALID_DATASET_NAMES, help="Dataset name e.g. iridia_af, afdb")
    parser.add_argument("--model-type", type=str, default="cnn", choices=VALID_MODELS, help="Model e.g. cnn")
    parser.add_argument("--file-indices", type=int, nargs="+", default=None, help="Indices of files to process (0-based). If not specified, processes all files")
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    parser.add_argument("--cv", action="store_true", help="Use cross-validation (5 folds)")
    args = parser.parse_args()

    run_optimisation_function(args.dataset_name, args.model_type, args.file_indices, args.resume, args.cv)