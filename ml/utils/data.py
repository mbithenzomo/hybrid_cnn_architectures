import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from load_config import load_config
from ml.utils.load_dataset import PredictionDataset

config = load_config()
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
DATASET_DIR = config["paths"]["proc_data_dir"]
K_FOLDS = config["cv"]["k_folds"]
LEADS = config["data"]["leads"]
NUM_WORKERS = config["data"]["num_workers"]
PATIENCE = config["training"]["patience"]
RANDOM_SEED = config["random_seed"]
VALID_DATASET_NAMES = config["valid_dataset_names"]

torch.manual_seed(RANDOM_SEED)

def get_class_distribution(filename, dataset_name):
    """
    Get the distribution of positive and negative classes from a dataset.
    
    Args:
        filename (str): Name of the dataset CSV file
        dataset_name (str): Dataset name e.g. "iridia_af"
    
    Returns:
        dict: Dictionary with 'positive', 'total', 'percentage', and 'formatted' keys
    """
    df = get_df(filename=filename, dataset_name=dataset_name)

    label_column = "label"  
    total_samples = len(df)
    positive_samples = (df[label_column] == 1).sum()
    negative_samples = total_samples - positive_samples
    positive_percentage = (positive_samples / total_samples * 100) if total_samples > 0 else 0
    
    formatted_str = f"{positive_samples}/{total_samples} ({positive_percentage:.1f}%)"
    
    return {
        "positive": positive_samples, 
        "negative": negative_samples, 
        "total": total_samples, 
        "percentage": positive_percentage, 
        "formatted": formatted_str
    }

def get_data_loaders(filename, dataset_name, cv_mode, num_runs=1, balance_train=True, balance_all=False):
    """
    
    Args:
        filename: Name of the dataset file
        dataset_name (str): Dataset name e.g. "iridia_af"
        cv_mode (str):
            - "kfold": Uses k-fold cross-validation.
            - "loocv": Uses leave-one-out cross-validation.
            - "single": Uses a single train-validation-test split (no cross-validation).
            - "final": Trains on the entire dataset without a test set (for external validation).
        num_runs (int): Number of runs on single split for stability testing (only used when cv_mode="single").
            - If num_runs > 1: trains multiple times with different random seeds on the same data split.
        balance_train (bool): If True, use WeightedRandomSampler to balance training data
        balance_all (bool): If True, use WeightedRandomSampler to balance train, val, and test sets. Defaults to False.
    
    Yields:
        tuple: (train_loader, val_loader, test_loader, fold_num, patient_list) for each fold
    """
    
    df = get_df(filename, dataset_name)
    patients = df["patient_id"].unique()

    if dataset_name == "afdb":
        before = len(patients)
        af_episodes = df.groupby("patient_id")["label"].mean()

        # remove persistent AF
        non_persistent_patients = af_episodes[af_episodes < 0.9].index
        patients = df[df["patient_id"].isin(non_persistent_patients)]["patient_id"].unique()
        print(f"Excluded records with persistent AF throughout record. Patient count: {before} -> {len(patients)}")

    if cv_mode == "final":
        print(f"\n{'='*70}")
        print(f"Final training on all data (no test set)")
        print(f"{'='*70}\n")
        train_patients = patients
        test_patients = np.array([])  
        val_patients = np.array([])   
        patient_list = [train_patients, val_patients, test_patients]
        train_loader, val_loader, test_loader = get_loaders(df=df, dataset_name=dataset_name, patient_list=patient_list, balance_train=balance_train, balance_all=balance_all)
        yield train_loader, val_loader, test_loader, None, patient_list

    elif cv_mode == "loocv":
        for i, test_patient in enumerate(patients, 1):
            print(f"\n{'='*70}")
            print(f"Patient {i}/{len(patients)}")
            print(f"{'='*70}\n")
            train_val_patients = patients[patients != test_patient]
            test_patients = np.array([test_patient])
            train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=RANDOM_SEED + i)
            patient_list = [train_patients, val_patients, test_patients]
            train_loader, val_loader, test_loader = get_loaders(df=df, dataset_name=dataset_name, patient_list=patient_list, balance_train=balance_train, balance_all=balance_all)
            yield train_loader, val_loader, test_loader, i, patient_list

    elif cv_mode == "kfold":
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(patients), 1):
            print(f"\n{'='*70}")
            print(f"Fold {fold}/{K_FOLDS}")
            print(f"{'='*70}\n")
            train_val_patients = patients[train_val_idx]
            test_patients = patients[test_idx]
            train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=RANDOM_SEED + fold)
            patient_list = [train_patients, val_patients, test_patients]
            train_loader, val_loader, test_loader = get_loaders(df=df, dataset_name=dataset_name, patient_list=patient_list, balance_train=balance_train, balance_all=balance_all)
            yield train_loader, val_loader, test_loader, fold, patient_list
    
    else:
        train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=RANDOM_SEED)
        train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=RANDOM_SEED)
        patient_list = [train_patients, val_patients, test_patients]
        train_loader, val_loader, test_loader = get_loaders(df=df, dataset_name=dataset_name, patient_list=patient_list, balance_train=balance_train, balance_all=balance_all)
        for run in range(1, num_runs + 1):
            yield train_loader, val_loader, test_loader, run, patient_list

def get_df(filename, dataset_name):
    """
    Get the dataset file from the filename.
    
    Args:
        filename (str): Name of the dataset CSV file
        dataset_name (str): Dataset name e.g. "iridia_af"
    
    Returns:   
        pd.DataFrame: DataFrame containing the dataset
    """
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(f"Name '{dataset_name}' is not valid. Valid dataset names: {VALID_DATASET_NAMES}")
    
    file_path = DATASET_DIR/dataset_name/filename
    df = pd.read_csv(file_path)

    return df

def get_filenames(dataset_name, pattern="dataset_*.csv"):
    """
    Get all dataset filenames from the datasets directory.
    
    Args:
        dataset_name (str): Dataset name e.g. "iridia_af"
        pattern (str): Glob pattern to match dataset files (default: "dataset_*.csv")
    
    Returns:
        list: List of dataset filenames (just the filenames, not full paths)
    """
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(f"Name '{dataset_name}' is not valid. Valid dataset names: {VALID_DATASET_NAMES}")

    file_path = DATASET_DIR/dataset_name

    dataset_files = sorted(file_path.glob(pattern))
    filenames = [f.name for f in dataset_files]

    def get_sort_key(filename):
        params = get_parameters(filename, dataset_name)
        return (params["prediction_horizon"], 
                params["input_window_size"], 
                params["step_size"],
                params["target_window_size"])
    
    sorted_filenames = sorted(filenames, key=get_sort_key)
    
    print(f"Found {len(sorted_filenames)} datasets in path {file_path}")
    
    return sorted_filenames

def get_parameters(filename, dataset_name):
    """
    Parse the parameters from the filename.

    Args:
        filename: Name of the CSV file
    
    Returns:
        dict: Dictionary containing parsed parameters
    """
    params_str = filename.replace("dataset_", "").replace(".csv", "")
    parts = params_str.split("_")
    
    params = {}
    for part in parts:
        if part.startswith("hor"):
            params["prediction_horizon"] = float(part.replace("hor", ""))
        elif part.startswith("inp"):
            params["input_window_size"] = float(part.replace("inp", ""))
        elif part.startswith("ste"):
            params["step_size"] = float(part.replace("ste", ""))
        elif part.startswith("tar"):
            params["target_window_size"] = float(part.replace("tar", ""))
    
    if dataset_name == "iridia_af":
        sampling_rate = 200
    elif dataset_name == "afdb":
        sampling_rate = 250
    elif dataset_name in ["nsrdb", "ltafdb"]:
        sampling_rate = 128
    
    params.update({
        "sampling_rate": sampling_rate
    })

    return params

def get_loaders(df, dataset_name, patient_list, balance_train, balance_all=False):
        """ 
        Get train, val, test loaders for a single split 

        Args:
            df (pd.DataFrame): DataFrame containing the dataset
            dataset_name (str): Dataset name e.g. "iridia_af"
            patient_list (list): [train_patients, val_patients, test_patients]
            balance_train (bool): If True, use WeightedRandomSampler to balance training data
            balance_all (bool): If True, use WeightedRandomSampler to balance train, val, and test sets. Defaults to False.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """

        # train loader
        train_df = df[df["patient_id"].isin(patient_list[0])]
        train_dataset = PredictionDataset(df=train_df, dataset_name=dataset_name, leads=LEADS)
        train_labels = train_df["label"].values

        class_counts = np.bincount(train_labels.astype(int))
        af_count = class_counts[1] if len(class_counts) > 1 else 0
        no_af_count = class_counts[0] if len(class_counts) > 0 else 0
        total = len(train_labels)
        af_percentage = 100 * af_count / total if total > 0 else 0

        print(f"Raw training set distribution: AF {af_count:,}; No AF {no_af_count:,} ({af_percentage:.2f}% AF)")

        if balance_train or balance_all:
            class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
            sample_weights = [class_weights[int(label)] for label in train_labels]   
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)   

            print(f"WeightedRandomSampler enabled for training: {len(sample_weights):,} samples per epoch")
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4)

        else:
            print(f"WeightedRandomSampler NOT enabled for training")
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4)
            
        if (len(patient_list[1]) == 0) or (len(patient_list[2]) == 0):
            return train_loader, None, None
        
        # val loader
        val_df = df[df["patient_id"].isin(patient_list[1])]
        val_dataset = PredictionDataset(df=val_df, dataset_name=dataset_name, leads=LEADS)

        if balance_all:
            val_labels = val_df["label"].values
            val_class_weights = compute_class_weight("balanced", classes=np.unique(val_labels), y=val_labels)
            val_sample_weights = [val_class_weights[int(label)] for label in val_labels]
            val_sampler = WeightedRandomSampler(weights=val_sample_weights, num_samples=len(val_dataset), replacement=True)
            
            print(f"WeightedRandomSampler enabled for validation: {len(val_sample_weights):,} samples")
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4
            )
    
        # test loader       
        test_df = df[df["patient_id"].isin(patient_list[2])]
        test_dataset = PredictionDataset(df=test_df, dataset_name=dataset_name, leads=LEADS)

        if balance_all:
            test_labels = test_df["label"].values
            test_class_weights = compute_class_weight("balanced", classes=np.unique(test_labels), y=test_labels)
            test_sample_weights = [test_class_weights[int(label)] for label in test_labels]
            test_sampler = WeightedRandomSampler(weights=test_sample_weights, num_samples=len(test_dataset), replacement=True)
            
            print(f"WeightedRandomSampler enabled for testing: {len(test_sample_weights):,} samples")
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4
            )

        if balance_all:
            print(f"  All sets will see approximately 50/50 distribution\n")

        return train_loader, val_loader, test_loader

def worker_init_fn(worker_id):
    """
    Initialise random seeds for DataLoader workers to ensure reproducibility
    """
    np.random.seed(RANDOM_SEED + worker_id)
    random.seed(RANDOM_SEED + worker_id)