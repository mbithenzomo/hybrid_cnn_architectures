from pathlib import Path

import h5py
import numpy as np
import torch
import wfdb
from scipy import signal
from torch.utils.data import Dataset

from data.preprocessing.utils.ecg_filtering import apply_butter_bandpass_filter
from load_config import load_config

config = load_config()
RAW_DATA_DIR = config["paths"]["raw_data_dir"]


class PredictionDataset(Dataset):

    def __init__(self, df, dataset_name, leads=[0, 1]):
        """
        Initialise the class by loading windows from ECG files. Each sample consists of a fixed-size ECG window and a corresponding binary label.

        Args:
            df (pd.DataFrame): DataFrame containing window metadata
            dataset_name (str): Dataset name e.g. "iridia_af"
            leads (list): Lead indices to extract from the ECG data (e.g., [0] for lead I, [1] for lead II, and [0, 1] for both leads).
        """
        for l in leads:
            if l not in [0, 1]:
                raise ValueError(f"Invalid lead index {l}. Must be 0 or 1.")

        self.df = df
        self.dataset_name = dataset_name

        self.leads = sorted(list(set(leads)))

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of target windows in the dataset.
        """
        return len(self.df)

    def _get_h5(self, path):
        """
        Return an open, cached h5py file handle for `path`, opening it once
        per worker process on first access.

        The cache is created lazily (not in __init__) because h5py handles are
        not fork-safe: __init__ runs in the main process before the DataLoader
        forks its workers, so opening there would share a handle across
        processes. Building the cache on first __getitem__ access keeps each
        handle private to the worker that opened it. With persistent_workers=True
        the cache survives across epochs, so each record file is opened once for
        the whole run instead of once per sample.
        """
        if not hasattr(self, "_h5_cache"):
            self._h5_cache = {}
        key = str(path)
        f = self._h5_cache.get(key)
        if f is None:
            try:
                f = h5py.File(path, "r")
            except OSError as e:
                raise OSError(f"Failed to open HDF5 record {path}: {e}") from e
            self._h5_cache[key] = f
        return f

    def __getitem__(self, idx):
        """
        Load and return a single ECG window and its label.

        Args:
            idx (int): Index of the sample to retrieve (0 to len(dataset)-1).

        Returns:
            tuple: A tuple containing:
                - ecg_data (torch.Tensor): ECG window of shape (1, window_size)
                    Single-lead ECG data as float32
                - label (torch.Tensor): Binary label of shape (1,)
                    Classification label as float32
        """
        sample = self.df.iloc[idx]

        if self.dataset_name == "combined":
            source = sample.dataset
        else:
            source = self.dataset_name

        if source == "iridia_af":

            record_path = RAW_DATA_DIR / source / sample.file
            f = self._get_h5(record_path)

            key = list(f.keys())[0]
            if self.leads == [0]:
                # extract lead I and add channel dimension
                ecg_data = f[key][sample.start_idx:sample.end_idx, 0]
                ecg_data = ecg_data[:, np.newaxis]
            elif self.leads == [1]:
                # extract lead II and add channel dimension
                ecg_data = f[key][sample.start_idx:sample.end_idx, 1]
                ecg_data = ecg_data[:, np.newaxis]
            else:
                # extract both leads; no need to add channel dimension
                ecg_data = f[key][sample.start_idx:sample.end_idx, :2]

            ecg_data = apply_butter_bandpass_filter(ecg_data, 200)
        else:

            full_path = RAW_DATA_DIR / source / sample.file
            record_path = str(full_path).replace(".dat", "").replace(".hea", "").replace(".atr", "")
            record = wfdb.rdrecord(str(record_path), sampfrom=sample.start_idx, sampto=sample.end_idx)
            ecg_window = record.p_signal
            frequency = record.fs
            filtered_ecg = apply_butter_bandpass_filter(ecg_window, frequency)

            if self.leads == [0]:
                ecg_data = filtered_ecg[:, 0:1]

            elif self.leads == [1]:
                if filtered_ecg.shape[1] < 2:
                    raise ValueError(f"Lead 1 requested but only {filtered_ecg.shape[1]} leads available in {sample.file}")
                ecg_data = filtered_ecg[:, 1:2]

            else:
                if filtered_ecg.shape[1] < 2:
                    raise ValueError(f"Two leads requested but only {filtered_ecg.shape[1]} leads available in {sample.file}")
                ecg_data = filtered_ecg[:, :2]

        # convert to tensor and add lead dimension
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
        ecg_data = ecg_data.transpose(0, 1)  # shape: (n_leads, window_size)

        # resample to 200 Hz for combined dataset
        if self.dataset_name == "combined":
            if source != "iridia_af":
                if source == "afdb":
                    current_fs = 250
                elif source in ["nsrdb", "afpdb"]:
                    current_fs = 128

                target_fs = 200
                if current_fs != target_fs:
                    num_samples = int(ecg_data.shape[1] * target_fs / current_fs)
                    ecg_data = torch.tensor(
                        signal.resample(ecg_data.numpy(), num_samples, axis=1),
                        dtype=torch.float32
                    )

        # convert label to tensor and add dimension
        label = torch.tensor(sample.label, dtype=torch.float32)
        label = label.unsqueeze(0)  # shape: (1,)

        return ecg_data, label