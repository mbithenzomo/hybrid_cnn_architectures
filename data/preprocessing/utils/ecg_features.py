import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(filtered_ecg, frequency):
    """
    Detect R peaks in the ECG signal
    
    Parameters:
    -----------
    filtered_ecg : array_like
        Preprocessed ECG signal
    frequency : float
        Sampling frequency in Hz
        
    Returns:
    --------
    array_like
        Indices of R peaks
    """
    peaks, properties = find_peaks(filtered_ecg, height = 0.5 * max(filtered_ecg), distance = 0.5 * frequency)
    return peaks


def detect_p_wave_pr_segment(filtered_ecg, r_peaks, frequency):
    """
    Detect P-wave and PR segment based on R-peak positions
    
    Parameters:
    -----------
    filtered_ecg (array_like): Filtered ECG signal
    r_peaks (array_like): Indices of R-peaks
    frequency (float): Sampling frequency in Hz
        
    Returns:
    --------
    tuple
        (p_peaks, end_indices)
        p_peaks: array_like; indices of P-wave peaks
        end_indices: array_like; indices of the end of each PR segment
    """
    p_peaks = []
    end_indices = []
    
    # for each R peak after the first one
    for i in range(1, len(r_peaks)):
        # define PR interval search window
        rr_interval = r_peaks[i] - r_peaks[i-1]
        
        # search window: from previous R peak + 70% of RR interval to current R peak - 30 ms
        start_idx = r_peaks[i-1] + int(0.7 * rr_interval)
        end_idx = r_peaks[i] - int(0.03 * frequency) 
        
        # ensure indices are within bounds
        if start_idx >= end_idx or start_idx < 0 or end_idx >= len(filtered_ecg):
            continue
        
        # extract the PR segment
        pr_segment = filtered_ecg[start_idx:end_idx]
        
        # find the P-wave peak (the maximum in this segment)
        if len(pr_segment) > 0:
            p_idx = np.argmax(pr_segment) + start_idx
            p_peaks.append(p_idx)
            end_indices.append(end_idx)

    return (np.array(p_peaks), np.array(end_indices))

def get_pr_features(filtered_ecg, frequency, threshold=0.1):
    """
    Get P-wave and PR interval durations and boundaries by detecting onset and offset points
    
    Parameters:
    -----------
    filtered_ecg (array_like): Filtered ECG signal
    frequency (float): Sampling frequency in Hz
    threshold (float): Percentage of peak amplitude to use as threshold
        
    Returns:
    --------
    dict:
        p_wave_durations: List of P-wave durations in milliseconds
        p_wave_boundaries: List of tuples (onset, peak, offset) for each P-wave
        pr_interval_durations: List of PR interval durations in milliseconds
        pr_interval_boundaries: List of tuples (beginning, end) for each PR interval
    """

    # detect R peaks
    r_peaks = detect_r_peaks(filtered_ecg, frequency)
    
    # detect P-peaks and end of PR segments
    p_peaks, end_indices = detect_p_wave_pr_segment(filtered_ecg, r_peaks, frequency)

    p_wave_durations = []
    p_wave_boundaries = []
    pr_interval_durations = [] 
    pr_interval_boundaries = []
    
    # define search window size 
    search_window = int(0.15 * frequency)
    
    for p_peak, end_idx in zip(p_peaks, end_indices):
        p_amplitude = filtered_ecg[p_peak]
        threshold_ = threshold * p_amplitude

        # estimate baseline
        baseline_segment = filtered_ecg[max(0, p_peak - search_window):min(len(filtered_ecg), p_peak + search_window)]
        baseline = np.median(baseline_segment)
        
        # find onset - go backward until signal crosses threshold
        onset = p_peak
        for i in range(p_peak, max(0, p_peak - search_window), -1):
            # Check if signal crosses the threshold relative to baseline
            if abs(filtered_ecg[i] - baseline) < threshold_:
                onset = i
                break
        # subtract 5 ms to ensure we are not cutting the P-wave
        onset = max(0, onset - int(0.005 * frequency))

        # find offset - go forward until signal crosses threshold
        offset = p_peak
        for i in range(p_peak, min(len(filtered_ecg), p_peak + search_window)):
            if abs(filtered_ecg[i] - baseline) < threshold_:
                offset = i
                break
        # add 5 ms to ensure we are not cutting the P-wave
        offset = min(len(filtered_ecg), offset + int(0.005 * frequency))

        # get P-wave duration in ms
        p_wave_duration = (offset - onset) * 1000 / frequency
        p_wave_durations.append(p_wave_duration)
        p_wave_boundaries.append((onset, p_peak, offset))

        # get PR interval duration in ms: p-wave + PR segment
        pr_interval_duration = (end_idx - onset) * 1000 / frequency
        pr_interval_durations.append(pr_interval_duration)
        pr_interval_boundaries.append((onset, end_idx))
    
    return {
        "p_wave_durations": p_wave_durations,
        "p_wave_boundaries": p_wave_boundaries,
        "pr_interval_durations": pr_interval_durations,
        "pr_interval_boundaries": pr_interval_boundaries
    }

def get_rr_features(filtered_ecg, frequency):
    """
    Get RR interval durations and boundaries
    
    Parameters:
    -----------
    filtered_ecg (array_like): Filtered ECG signal
    frequency (float): Sampling frequency in Hz
    
    Returns:
    --------
    dict:
        rr_interval_durations: List of RR interval durations in milliseconds
    """

    r_peaks = detect_r_peaks(filtered_ecg, frequency)
    if len(r_peaks) < 2:
        return {
            "rr_interval_durations": np.nan,
            "rr_interval_boundaries": np.nan
        }

    rr_intervals = []
    rr_interval_boundaries = []

    for i in range(1, len(r_peaks)):
        rr_interval = r_peaks[i] - r_peaks[i-1]
        rr_intervals.append(rr_interval)
        rr_interval_boundaries.append((r_peaks[i-1], r_peaks[i]))
    
    rr_interval_durations = np.array(rr_intervals) * 1000 / frequency
    
    return {
        "rr_interval_durations": rr_interval_durations.tolist(),
        "rr_interval_boundaries": rr_interval_boundaries
    }

def visualise_waves_intervals(raw_ecg, filtered_ecg, p_wave_boundaries, pr_interval_boundaries, rr_interval_boundaries, frequency):
    """
    Visualise ECG with P-waves, PR intervals, and RR intervals annotated.
    
    Parameters:
    -----------
    raw_ecg : array_like
        Original ECG signal
    filtered_ecg : array_like
        Filtered ECG signal
    p_wave_boundaries : list
        List of tuples (onset, peak, offset) for each P-wave
    pr_interval_boundaries : list
        List of tuples (beginning, end) for each PR interval
    rr_interval_boundaries : list
        List of tuples (r_peak1, r_peak2) for each RR interval
    frequency : float
        Sampling frequency in Hz
    """
    time = np.arange(len(raw_ecg)) / frequency
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 5), sharex=True)
    
    # raw ECG
    ax1.plot(time, raw_ecg, "b")
    ax1.set_title("Original ECG")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    
    # filtered ECG with annotations
    ax2.plot(time, filtered_ecg, "b")

    # annotate RR intervals
    for i, (r_peak1, r_peak2) in enumerate(rr_interval_boundaries):
        rr_duration = (r_peak2 - r_peak1) * 1000 / frequency
        
        # mark R-peaks
        ax2.plot(time[r_peak1], filtered_ecg[r_peak1], "r*", markersize=12)
        ax2.plot(time[r_peak2], filtered_ecg[r_peak2], "r*", markersize=12)
        
        # annotate RR interval
        mid_point = (r_peak1 + r_peak2) // 2
        ax2.annotate(f"RR: {rr_duration:.0f} ms", 
                    xy=(time[mid_point], filtered_ecg[mid_point]),
                    xytext=(0, -20), 
                    textcoords="offset points",
                    ha="center",
                    color="red",
                    fontsize=9)
    
    # annotate each P-wave and PR interval
    for i, (onset, peak, offset) in enumerate(p_wave_boundaries):
        p_wave_duration = (offset - onset) * 1000 / frequency 
        ax2.axvline(x=time[onset], color="g", linestyle="--", alpha=0.7)
        ax2.axvline(x=time[offset], color="r", linestyle="--", alpha=0.7)
        ax2.plot(time[peak], filtered_ecg[peak], "mo")
        ax2.annotate(f"{p_wave_duration:.1f} ms", 
                    xy=(time[peak], filtered_ecg[peak]),
                    xytext=(0, 10), 
                    textcoords="offset points",
                    ha="center")
    
    for i, (beginning, end) in enumerate(pr_interval_boundaries):
        pr_interval_duration = (end - beginning) * 1000 / frequency
        ax2.axvline(x=time[end], color="cyan", linestyle="--", alpha=0.7)
        ax2.annotate(f"{pr_interval_duration:.1f} ms", 
                    xy=(time[beginning], filtered_ecg[beginning]),
                    xytext=(0, 10), 
                    textcoords="offset points",
                    ha="center")

    ax2.set_title("Filtered ECG with P-wave and PR interval annotations")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # zoom in to one heartbeat
    n_intervals = min(15, len(rr_interval_boundaries))  
    
    if n_intervals > 0:
        fig2, axes = plt.subplots(n_intervals, 1, figsize=(12, 3*n_intervals))
        if n_intervals == 1:
            axes = [axes]
        
        for i, (r_peak1, r_peak2) in enumerate(rr_interval_boundaries[:n_intervals]):
            padding = int(0.1 * frequency)
            start_idx = max(0, r_peak1 - padding)
            end_idx = min(len(filtered_ecg), r_peak2 + padding)         
            segment_time = time[start_idx:end_idx]
            segment = filtered_ecg[start_idx:end_idx]
            axes[i].plot(segment_time, segment, "b")

            # R-peak annotations
            axes[i].plot(time[r_peak1], filtered_ecg[r_peak1], "r*", markersize=15, label="R-peak")
            axes[i].plot(time[r_peak2], filtered_ecg[r_peak2], "r*", markersize=15)
            
            # RR interval annotations
            rr_duration = (r_peak2 - r_peak1) * 1000 / frequency
            axes[i].plot([time[r_peak1], time[r_peak2]], 
                       [filtered_ecg[r_peak1], filtered_ecg[r_peak2]], 
                       'r--', alpha=0.5, linewidth=2, label=f"RR interval: {rr_duration:.0f} ms")

            # P-wave annotations
            for onset, peak, offset in p_wave_boundaries:
                if r_peak1 <= onset <= r_peak2 or r_peak1 <= peak <= r_peak2:
                    p_wave_duration = (offset - onset) * 1000 / frequency
                    axes[i].axvline(x=time[onset], color="g", linestyle="-", alpha=0.7, label="P-wave onset")
                    axes[i].axvline(x=time[peak], color="m", linestyle="-", alpha=0.7)
                    axes[i].axvline(x=time[offset], color="r", linestyle="-", alpha=0.7, label="P-wave offset")
                    axes[i].plot(time[peak], filtered_ecg[peak], "mo", markersize=8)
                    
                    axes[i].fill_between(time[onset:offset+1], filtered_ecg[onset:offset+1], 
                                       color="yellow", alpha=0.3, label="P-wave")
                    
                    axes[i].text(time[peak], filtered_ecg[peak] + 0.1, 
                               f"P: {p_wave_duration:.1f} ms", 
                               ha="center", fontsize=9)
            
            # PR segment annotations
            for beginning, end in pr_interval_boundaries:
                if r_peak1 <= beginning <= r_peak2 or r_peak1 <= end <= r_peak2:
                    pr_interval_duration = (end - beginning) * 1000 / frequency
                    axes[i].axvline(x=time[end], color="cyan", linestyle="--", alpha=0.7, label="PR segment end")
                    
                    axes[i].fill_between(time[beginning:end+1], filtered_ecg[beginning:end+1], 
                                       color="lightblue", alpha=0.2, label="PR interval")
                    
                    axes[i].text(time[(beginning + end)//2], filtered_ecg[beginning] - 0.1, 
                               f"PR: {pr_interval_duration:.1f} ms", 
                               ha="center", color="cyan", fontsize=9)
            
            axes[i].set_title(f"RR Interval {i+1}: {rr_duration:.0f} ms")


            handles, labels = axes[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[i].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
            axes[i].grid(True)
            axes[i].set_ylabel("Amplitude")
            
            # add x-label to last subplot only
            if i == n_intervals - 1:
                axes[i].set_xlabel("Time (s)")
        
        plt.tight_layout()
    
    plt.show()

