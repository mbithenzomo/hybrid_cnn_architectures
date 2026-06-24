import atexit
import os
import traceback
import sys
from datetime import datetime

dataset_logs = "./ml/outputs/dataset_logs" 
os.makedirs(dataset_logs, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"{dataset_logs}/{timestamp}.log", "w", buffering=1)
atexit.register(log_file.close)
sys.stdout = log_file
sys.stderr = log_file

import numpy as np
import pandas as pd
import wfdb

from data.preprocessing.create_record import create_record
from load_config import load_config

config = load_config()
DATASET_DIR = config["paths"]["proc_data_dir"]
RAW_DATA_DIR = config["paths"]["raw_data_dir"]
VALID_DATASET_NAMES = config["valid_dataset_names"]


def create_prediction_dataset(prediction_dict, dataset_dict):
    """
    Create a CSV dataset for AF prediction using sliding windows.
    
    Args:
        prediction_dict (dict): Dictionary containing prediction parameters:
            prediction_horizon (int): How far into the future to predict (minutes)
            input_window_size (int): Size of the input ECG window (minutes)
            step_size (float): Controls overlap between consecutive windows (minutes)
            target_window_size (int): Size of the target window (minutes)

        dataset_dict (dict): Dictionary containing ECG dataset parameters:
            dataset_name (str): Name of the dataset
            records_path (str): Directory path containing ECG record files

    Returns:
        dataset_path (str): Path to the created dataset CSV file
    """
    dataset_name = dataset_dict["dataset_name"]
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(f"Name '{dataset_name}' is not valid. Valid dataset names: {VALID_DATASET_NAMES}")
    
    records_path = dataset_dict["records_path"]
    output_path = DATASET_DIR / dataset_name

    if dataset_name == "iridia_af":
        sampling_rate = 200
        record_ids = [f.name for f in records_path.iterdir() if f.is_dir()]
    else:
        hea_ids = {f.stem for f in records_path.glob("*.hea")}
        dat_ids = {f.stem for f in records_path.glob("*.dat")}
        atr_ids = {f.stem for f in records_path.glob("*.atr")}
        record_ids = list(hea_ids & dat_ids & atr_ids)
        
        sample_record = wfdb.rdrecord(str(records_path / record_ids[0]))
        sampling_rate = sample_record.fs
    
    record_ids = sorted(record_ids)

    # convert time parameters to samples
    prediction_horizon = int(prediction_dict["prediction_horizon"] * 60 * sampling_rate)
    input_window = int(prediction_dict["input_window_size"] * 60 * sampling_rate)
    step_size = int(prediction_dict["step_size"] * 60 * sampling_rate)
    target_window = int(prediction_dict["target_window_size"] * 60 * sampling_rate)
    overlap_pct = (1 - (step_size / input_window)) * 100

    print(f"Parameters:")
    print(f"  Input window: {prediction_dict['input_window_size']} mins")
    print(f"  Prediction horizon: {prediction_dict['prediction_horizon']} mins")
    print(f"  Step size: {prediction_dict['step_size']} mins")
    print(f"  Target window: {prediction_dict['target_window_size']} mins")
    print(f"  Window overlap: {overlap_pct}%")
    print(f"{'='*70}\n")

    list_windows = []
    total_windows = 0
    positive_windows = 0
    negative_windows = 0
    skipped_files = 0

    required_length = input_window + prediction_horizon + target_window
    required_length_mins = required_length / (60 * sampling_rate)

    for record_id in record_ids:
        record = create_record(record_id, records_path, dataset_name)
        record.load_ecg()

        if dataset_name == "iridia_af":
            # IRIDIA-AF: list of arrays (one per day)
            ecg_data_list = record.ecg
            ecg_labels_list = record.ecg_labels
            ecg_files_list = record.ecg_files
        else:
            # WFD datasets: single array 
            ecg_data_list = [record.ecg]
            ecg_labels_list = [record.ecg_labels]
            ecg_files_list = [record.ecg_files[0]]

        # process each file/day
        for idx, (ecg_data, ecg_labels, ecg_file) in enumerate(
            zip(ecg_data_list, ecg_labels_list, ecg_files_list)
        ):

            length = ecg_data.shape[0]
            length_mins = length / (60 * sampling_rate)

            # check if file is long enough
            if length < required_length:
                skipped_files += 1
                print(f"Warning: File {ecg_file} for record {record_id} is too short. "
                      f"Only {length_mins:.2f} minutes available, need at least {required_length_mins:.2f} minutes. "
                      f"Skipping this file.")
                continue

            max_start_idx = length - required_length
            windows_this_file = 0

            # create sliding windows
            for start_idx in range(0, max_start_idx, step_size):
                end_idx = start_idx + input_window
                target_start_idx = end_idx + prediction_horizon
                target_end_idx = target_start_idx + target_window

                target_labels = ecg_labels[target_start_idx:target_end_idx]
                label = 1 if np.sum(target_labels) > 0 else 0

                full_path = str(ecg_file)
                relative_path = full_path.split(f"{dataset_name}/", 1)[1]

                patient_id = record.metadata.patient_id if dataset_name == "iridia_af" else record_id

                prediction_window = {
                    "patient_id": patient_id,
                    "file": relative_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "prediction_horizon_samples": prediction_horizon,
                    "prediction_horizon_minutes": prediction_dict["prediction_horizon"],
                    "target_start_idx": target_start_idx,
                    "target_end_idx": target_end_idx,
                    "label": label
                }

                list_windows.append(prediction_window)
                windows_this_file += 1
                total_windows += 1
                if label == 1:
                    positive_windows += 1
                else:
                    negative_windows += 1

    dataset = pd.DataFrame(list_windows)
    filename = (f"dataset_"
                f"hor{prediction_dict['prediction_horizon']}_"
                f"inp{prediction_dict['input_window_size']}_"
                f"ste{prediction_dict['step_size']}_"
                f"tar{prediction_dict['target_window_size']}.csv")

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path = output_path / filename
    dataset.to_csv(dataset_path, index=False)

    print(f"\n{'='*70}")
    print(f"Dataset Creation Complete")
    print(f"{'='*70}")
    print(f"Output file: {dataset_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total windows: {total_windows:,}")
    if total_windows > 0:
        print(f"  AF positive windows: {positive_windows:,} ({100 * positive_windows / total_windows:.2f}%)")
        print(f"  AF negative windows: {negative_windows:,} ({100 * negative_windows / total_windows:.2f}%)")
    print(f"  Skipped files (too short): {skipped_files}")
    print(f"{'='*70}\n")

    return dataset_path

def create_all_datasets(configurations, dataset_name, input_window_size=5.0, target_window_size=0.5, overlap=0.7):
    """
    Create multiple datasets with varying parameters for experimentation.
    
    Args:
        configurations (list of dicts): Each dict contains prediction_horizon and input_window_size, some contain step size and target window size
        dataset_name (str): Name of the dataset
        input_window_size (float): Size of the input window in minutes (default: 5.0)
        target_window_size (float): Size of the target window in minutes (default: 0.5)
        overlap (float): Overlap between consecutive windows (default: 0.7 i.e. 70%)
    
    Returns:
        list: Paths to all created datasets
    """
    created_datasets = []
    total_configs = len(configurations)
    
    print(f"\n{'='*70}")
    print(f"Creating {total_configs} datasets with different parameter combinations")
    print(f"{'='*70}\n")
    
    config_num = 0

    records_path = RAW_DATA_DIR / dataset_name
    if dataset_name == "iridia_af":
        records_path = records_path / "records"
    
    for config in configurations:
        config_num += 1
        print(f"\n{'*'*30}")
        print(f"Creating Dataset [{config_num}/{total_configs}]")
        print(f"{'*'*30}")
        print(f"Dataset: {dataset_name}")

        prediction_horizon = config["horizon"]

        if "input" in config:
            input_window_size = config["input"]
        
        if "target" in config:
            target_window_size = config["target"]

        if "step" in config:
            step_size = config["step"]
        else:
            # calculate step size based on deafult 70% overlap
            # overlap = (1 - step_size/input_window)
            # therefore step_size = input_window * (1 - overlap)
            step_size = input_window_size * (1 - overlap)
            step_size = round(step_size, 2)
       
        prediction_dict = {
            "input_window_size": input_window_size,
            "prediction_horizon": prediction_horizon,
            "target_window_size": target_window_size,
            "step_size": step_size,
        }
        
        dataset_dict = {
            "dataset_name": dataset_name,
            "records_path": records_path
        }
        
        try:
            dataset_path = create_prediction_dataset(
                prediction_dict, 
                dataset_dict
            )
            created_datasets.append(dataset_path)
            print(f"  ✓ Success: {dataset_path}")
        except Exception as e:
            print(traceback.format_exc())
        
    print(f"\n{'='*70}")
    print(f"{len(created_datasets)}/{total_configs} datasets created.")
    print(f"{'='*70}\n")
    
    return created_datasets


if __name__ == "__main__":
    configurations = [

        # prediction horizon configs: 9 horizons
        # fixed parameters: input window = 5 mins, target window = 0.5 mins, overlap = 75%
        # {"horizon": 0.5}, # also Gregoire et al. (2022) & Gilon et al. (2020)
        {"horizon": 5.0}, # also Gilon et al. (2020)
        # {"horizon": 10.0},       
        {"horizon": 15.0},       
        {"horizon": 30.0},       
        {"horizon": 45.0},       
        {"horizon": 60.0},       

        # # SOTA comparison configs
        # # all values are in mins, rounded to 4dp where necessary

        # # Gregoire et al. (2025)
        # # target is based on "episodes lasting more than 5 min of AF"
        # # horizon is based on "60 min of normal sinus rhythm prior to AF onset"/"one-hour window of interest"
        # {"horizon": 60.0, "target": 5.0},       

        # # Li et al. (2025) (input is 10s, step is 5s)
        # {"horizon": 18.9, "input": 0.1667, "step": 0.0833}, 
        
        # # Gavidia et al. (2025) (step is 15s)
        # {"horizon": 32.5, "step": 0.25}, 
        
        # # Rooney et al. (2023)
        # # target is based on "we defined an AF episode of interest as any AF annotation ≥5 min long"
        # {"horizon": 7.5, "input": 3.0, "step": 0.5, "target": 5.0},
        # {"horizon": 15.0, "input": 3.0, "step": 0.5, "target": 5.0},
        # {"horizon": 30.0, "input": 3.0, "step": 0.5, "target": 5.0},
        # {"horizon": 60.0, "input": 3.0, "step": 0.5, "target": 5.0},

        # # Gilon et al. (2020)
        # {"horizon": 1.0}, 
        # {"horizon": 1.5}, 

        # # Chen et al. (2024)
        # # input is 100 RRI which is 1 min 40s at 60 bpm
        # # study overlap is 99% but using 70% due to computational constraints
        # {"horizon": 45.0, "input": 1.6667, "target": 0.5}, 

    ]
    
    for name in ["ltafdb"]:
        created_paths = create_all_datasets(configurations, dataset_name=name)  
        print(f"\nAll dataset paths:")
        for path in created_paths:
            print(f"  - {path}")