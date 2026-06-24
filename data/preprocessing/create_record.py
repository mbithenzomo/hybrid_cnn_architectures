# Adapted from:
# Original author: Cédric Gilon
# Source: https://github.com/cedricgilon/iridia-af/blob/main/iridia_af/record.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import hrvanalysis as hrv
import h5py
import numpy as np
import pandas as pd
import wfdb
from matplotlib import pyplot as plt

from load_config import load_config

config = load_config()
RAW_DATA_DIR = config["paths"]["raw_data_dir"]

def create_record(record_id, records_path, dataset_name):
    """
    Create a Record object for a given record ID
    
    Args:
        record_id (str): The record identifier
        records_path (str): Path to the record folder
        dataset_name (str): Name of the dataset e.g. "iridia_af

    Returns:
        Record: An instance of the Record class
    """
    if dataset_name == "iridia_af":
        records_path = Path(records_path, record_id)
    else:
        records_path = Path(records_path)
    record = Record(record_id, records_path, dataset_name)

    return record


class Record:
    def __init__(self, record_id, records_path, dataset_name):
        """
        Initialise a Record object for a given record ID
        
        Args:
            record_id (str): The record identifier
            records_path (str): Path to the record folder
            dataset_name (str): Name of the dataset e.g. "iridia_af
        """
        self.annotations = None
        self.dataset_name = dataset_name
        self.ecg = None
        self.ecg_labels = None
        self.ecg_labels_df = None
        self.record_id = record_id
        self.records_path = records_path

        if dataset_name == "iridia_af":

            metadata_path = RAW_DATA_DIR / "iridia_af/metadata.csv"
            metadata_df = pd.read_csv(metadata_path)
            metadata_record = (metadata_df[metadata_df["record_id"] == record_id])
            assert len(metadata_record) == 1
            metadata_record = metadata_record.values[0]
            self.metadata = RecordMetadata(*metadata_record)

            self.num_days = len(list(records_path.glob("*ecg_*.h5")))
            self.ecg_files = sorted(records_path.glob("*ecg_*.h5"))
            assert len(self.ecg_files) == self.num_days

        else:
            self.record = wfdb.rdrecord(records_path / record_id)
            if not (records_path / f"{record_id}.hea").exists():
                raise FileNotFoundError(f"Header (metadata) file not found: {record_id}.hea")
            self.metadata = self.__read_wfdb_metadata()
            self.num_days = 1 # there is only a single day per record
            self.ecg_files = [records_path / f"{record_id}.dat"] 

    
    def __read_wfdb_metadata(self):
        """
        Helper function to read metadata from WFDB files
        """        
        duration_seconds = self.record.sig_len / self.record.fs
        
        end_time = None
        if self.record.base_time:
            start_datetime = datetime.combine(datetime.today(), self.record.base_time)
            end_datetime = start_datetime + timedelta(seconds=duration_seconds)
            end_time = end_datetime.time()
        
        metadata = RecordMetadata(
            patient_id=None, # check for other datasets
            patient_sex=None,
            patient_age=None,
            record_id=self.record_id,
            record_date=self.record.base_date,
            record_start_time=self.record.base_time,
            record_end_time=end_time,
            record_timedelta=duration_seconds,
            record_files=1,
            record_seconds=duration_seconds,
            record_samples=self.record.sig_len
        )
        return metadata

    def load_ecg(self, clean_front=False):
        if self.dataset_name == "iridia_af":
            ecg_files = sorted(self.records_path.glob("*_ecg_*.h5"))
            self.ecg = [self.__read_ecg_file(ecg_file, clean_front) for ecg_file in ecg_files]
            self.__create_ecg_labels_iridia(clean_front)
            
        else:
            self.ecg = self.record.p_signal
            if clean_front:
                self.ecg = self.ecg[6000:]
            self._load_annotations_wfdb()
            self.__create_ecg_labels_wfdb(clean_front)

    def __read_ecg_file(self, ecg_file: Path, clean_front=False) -> np.ndarray:
        with h5py.File(ecg_file, "r") as f:
            key = list(f.keys())[0]
            ecg = f[key][:]
            if clean_front:
                ecg = ecg[6000:]
        return ecg

    def __create_ecg_labels_iridia(self, clean_front=False):
        ecg_labels = sorted(self.records_path.glob("*ecg_labels.csv"))
        self.ecg_labels_df = pd.read_csv(ecg_labels[0])
        len_ecg = [len(ecg) for ecg in self.ecg]

        start_day = self.ecg_labels_df["start_file_index"].unique()
        end_day = self.ecg_labels_df["end_file_index"].unique()
        days = np.unique(np.concatenate([start_day, end_day]))
        assert len(days) <= len(len_ecg)

        labels = [np.zeros(len_day_ecg) for len_day_ecg in len_ecg]

        for i, row in self.ecg_labels_df.iterrows():
            if row.start_file_index == row.end_file_index:
                labels[row.start_file_index][row.start_qrs_index:row.end_qrs_index] = 1
            else:
                labels[row.start_file_index][row.start_qrs_index:] = 1
                labels[row.end_file_index][:row.end_qrs_index] = 1
                if row.end_file_index - row.end_file_index > 1:
                    for day in range(row.start_file_index + 1, row.end_file_index):
                        labels[day][:] = 1
        if clean_front:
            labels = [label[6000:] for label in labels]
        self.ecg_labels = labels

    def _load_annotations_wfdb(self):
        """
        Load annotations from WFDB .atr file
        """
        self.annotations = wfdb.rdann(str(self.records_path / self.record_id), "atr")

        rhythm_changes = []
        for i, (sample, symbol, aux) in enumerate(
            zip(self.annotations.sample, 
                self.annotations.symbol, 
                self.annotations.aux_note)
        ):
            # check for change in rhythm and append annotation
            if symbol == "+" and aux:  
                rhythm_changes.append({
                    "sample": sample,
                    "time_seconds": sample / self.metadata.record_samples * self.metadata.record_seconds,
                    "rhythm": aux.strip('('),  
                })
        
        self.ecg_labels_df = pd.DataFrame(rhythm_changes)

    def __create_ecg_labels_wfdb(self, clean_front=False):
        """
        Create sample-by-sample labels for WFDB datasets 
        where 0 = NSR and 1 = AF
        """
        n_samples = len(self.ecg)
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(len(self.ecg_labels_df)):
            start_sample = self.ecg_labels_df.iloc[i]["sample"]
            rhythm = self.ecg_labels_df.iloc[i]["rhythm"]
            
            if clean_front:
                start_sample = max(0, start_sample - 6000)
            
            # find end sample (next rhythm change or end of record)
            if i + 1 < len(self.ecg_labels_df):
                end_sample = self.ecg_labels_df.iloc[i + 1]["sample"]
                if clean_front:
                    end_sample = max(0, end_sample - 6000)
            else:
                end_sample = n_samples
            
            if rhythm == "AFIB":
                labels[start_sample:end_sample] = 1
        
        self.ecg_labels = labels


@dataclass
class RecordMetadata:
    patient_id: str
    patient_sex: str
    patient_age: int
    record_id: str
    record_date: str
    record_start_time: str
    record_end_time: str
    record_timedelta: str
    record_files: int
    record_seconds: int
    record_samples: int

@dataclass
class ECGEvent:
    # start event
    start_datetime: str
    start_file_index: int
    start_qrs_index: int
    # end event
    end_datetime: str
    end_file_index: int
    end_qrs_index: int
    # duration
    af_duration: int
    nsr_duration: int
