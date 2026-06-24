import numpy as np
import pandas as pd
from scipy import stats

from load_config import load_config

config = load_config()
RANDOM_SEED = config["random_seed"]

def aggregate_metrics(metrics, confidence=0.95):
    """
    Aggregate metrics across all folds with confidence intervals.
    
    Args:
        metrics: List of metric dictionaries from each fold
        confidence: Confidence level for CI (default: 0.95)
    
    Returns:
        dict: Aggregated metrics with mean, std, and confidence intervals
    """
    metrics_df = pd.DataFrame(metrics)
    
    aggregated = {}
    total_cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    
    for metric in metrics_df.columns:
        if metric == "confusion_matrix":
            for _, row in metrics_df.iterrows():
                cm = row["confusion_matrix"]
                for key in total_cm:
                    total_cm[key] += cm[key]
            aggregated["confusion_matrix"] = total_cm
        else:
            scores = metrics_df[metric].dropna().values
            scores = np.array([s for s in scores if s is not None])

            if len(scores) == 0:
                aggregated[f"{metric}_mean"] = None
                aggregated[f"{metric}_std"] = None
                aggregated[f"{metric}_ci_lower"] = None
                aggregated[f"{metric}_ci_upper"] = None
                continue

            ci_stats = get_confidence_intervals(scores, confidence)

            aggregated[f"{metric}_mean"] = ci_stats["mean"]
            aggregated[f"{metric}_std"] = ci_stats["std"]

            if metric != "loss":
                if ci_stats["std"] == 0:
                    aggregated[f"{metric}_ci_lower"] = aggregated[f"{metric}_mean"]
                    aggregated[f"{metric}_ci_upper"] = aggregated[f"{metric}_mean"]
                else:
                    aggregated[f"{metric}_ci_lower"] = ci_stats["ci_lower"]
                    aggregated[f"{metric}_ci_upper"] = ci_stats["ci_upper"]

    aggregated["results"] = metrics
        
    return aggregated

def get_confidence_intervals(scores, confidence=0.95):
    """
    Get confidence interval for a list of scores for cross-validation.
    
    Args:
        scores: List or array of scores from each fold
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        dict: mean, std, ci_lower, ci_upper, sem
    """

    # calculate confidence interval using bootstrap method
    scores = np.array(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return {"mean": None, "std": None, "ci_lower": None, "ci_upper": None, "ci_range": None}
        
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    res = stats.bootstrap((scores,), np.mean, confidence_level=confidence, random_state=RANDOM_SEED)
    
    return {
        "mean": mean,
        "std": std,
        "ci_lower": res.confidence_interval.low,
        "ci_upper": res.confidence_interval.high,
        "ci_range": res.confidence_interval.high - res.confidence_interval.low,
    }