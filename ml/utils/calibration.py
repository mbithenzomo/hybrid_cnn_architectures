import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, PrecisionRecallDisplay, RocCurveDisplay, roc_auc_score

def convert_to_binary(y_true, threshold=0.5):
    y_true = np.asarray(y_true).flatten()
    return (y_true >= threshold).astype(int)

def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    y_true = convert_to_binary(y_true)
    y_pred_proba = np.array(y_pred_proba)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_pred_proba[mask].mean()
        bin_weight = mask.sum() / len(y_pred_proba)
        ece += bin_weight * np.abs(bin_acc - bin_conf)

    return ece

def apply_calibrator(calibrator, y_pred_proba):
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    return calibrator.transform(y_pred_proba)  

def fit_calibrator(y_pred_proba, y_true, method="platt"):
    """Fit on a held-out (validation) split only."""
    y_true = convert_to_binary(y_true)
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    if method == "platt":
        cal = LogisticRegression()
        cal.fit(y_pred_proba.reshape(-1, 1), y_true)
    elif method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(y_pred_proba, y_true)
    else:
        raise ValueError("method must be 'platt' or 'isotonic'")
    return apply_calibrator(cal, y_pred_proba), cal

def build_calibrated_dict(y_true, uncal_proba, calibrators, n_bins=10):
    out = {
        "Uncalibrated": (uncal_proba,
                         brier_score_loss(convert_to_binary(y_true), uncal_proba),
                         expected_calibration_error(y_true, uncal_proba, n_bins))
    }
    for label, key in (("Platt", "platt"), ("Isotonic", "isotonic")):
        p = apply_calibrator(calibrators[key], uncal_proba)
        out[label] = (p,
                      brier_score_loss(convert_to_binary(y_true), p),
                      expected_calibration_error(y_true, p, n_bins))
    return out

def show_plots(y_true, model, n_bins=10, calibrated_dict=None, plot="calibration"):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. Will be passed through convert_to_binary before use.
    model : str
        Model name, used in the plot title.
    n_bins : int, optional
        Number of bins to use in the calibration curve. Default is 10.
    calibrated_dict : dict, optional
        Dictionary mapping calibration method names to a tuple of
        (calibrated_probs, brier, ece), e.g.:
            {
                "Uncalibrated": (unc_probs, 0.34, 0.21),
                "Platt":        (platt_probs, 0.12, 0.05),
                "Isotonic":     (iso_probs, 0.10, 0.03),
            }
    plot : str, optional
        Plot(s) to display. One of  One of "calibration", "auroc", or "auprc".
        Default is "calibration".
    Effects
    -------
    Displays the figure inline.
    """
    y_true = convert_to_binary(y_true)
    blues = plt.get_cmap("Blues")

    def format_title(name, scores):
        if scores:
            brier, ece = scores
            return f"{name}\n(Brier: {brier:.4f} | ECE: {ece:.4f})"
        return name

    prob_list = []
    if calibrated_dict:
        cmap_list = [blues(0.4), blues(1.0), blues(0.7)]
        for idx, (name, (probs, brier, ece)) in enumerate(calibrated_dict.items()):
            prob_list.append((name, np.array(probs), cmap_list[idx % len(cmap_list)], (brier, ece)))

    show_calibration = plot in ("calibration", "all")
    show_auroc = plot in ("auroc", "all")
    show_auprc = plot in ("auprc", "all")        

    n_top_cols = sum([show_calibration, show_auroc, show_auprc]) 
    n_cols = max(n_top_cols, 1)
    n_rows = 1

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor("white")
    gs = GridSpec(n_rows, n_cols)

    top_col = 0

    if show_calibration:
        ax_cal = fig.add_subplot(gs[0, top_col])
        top_col += 1
        for name, y_pred_proba, color, scores in prob_list:
            CalibrationDisplay.from_predictions(
                y_true, y_pred_proba, n_bins=n_bins,
                name=format_title(name, scores), ax=ax_cal, color=color
            )
        ax_cal.grid()
        ax_cal.legend(loc="upper left")
        ax_cal.set_title(f"Calibration Plot — {model}")

    if show_auroc:
        ax_roc = fig.add_subplot(gs[0, top_col])
        top_col += 1
        for name, y_pred_proba, color, scores in prob_list:
            auroc = roc_auc_score(y_true, y_pred_proba)
            RocCurveDisplay.from_predictions(
                y_true, y_pred_proba, name=name, ax=ax_roc, color=color
            )
            ax_roc.lines[-1].set_label(f"{name} (AUROC = {auroc:.2f})")
        ax_roc.plot([0, 1], [0, 1], "k:", label="Random classifier")
        ax_roc.grid()
        ax_roc.legend(loc="lower right")
        ax_roc.set_title(f"ROC Curve — {model}")

    if show_auprc:
        ax_pr = fig.add_subplot(gs[0, top_col])
        top_col += 1
        prevalence = y_true.mean()
        for name, y_pred_proba, color, scores in prob_list:
            ap = average_precision_score(y_true, y_pred_proba)
            PrecisionRecallDisplay.from_predictions(
                y_true, y_pred_proba,
                name=name,
                ax=ax_pr, color=color,
                plot_chance_level=False
            )
            ax_pr.lines[-1].set_label(f"{name} (AUPRC = {ap:.2f})")
        ax_pr.axhline(y=prevalence, color="k", linestyle="--", label=f"Baseline")
        ax_pr.set_aspect("auto")
        ax_pr.grid()
        ax_pr.legend(loc="upper right")
        ax_pr.set_title(f"Precision-Recall Curve — {model}")

    return fig