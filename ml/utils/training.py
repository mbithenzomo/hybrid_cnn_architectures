import glob

import numpy as np
import torch

from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize_scalar

from load_config import load_config
from ml.utils.calibration import expected_calibration_error

config = load_config()
RANDOM_SEED = config["random_seed"]
WARMUP_START_FACTOR = 0.01

torch.manual_seed(RANDOM_SEED)

def configure_optimizers(model, learning_rate, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    return optimizer

@torch.no_grad()
def estimate_loss(model, device, dataset, criterion):
    model.eval()
    losses = []

    for x, y in dataset:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())

    return np.mean(losses)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def calc_fbeta(precision, recall, beta):
        scores = ((1 + beta**2) * precision[:-1] * recall[:-1]) / ((beta**2 * precision[:-1]) + recall[:-1])
        return np.nan_to_num(scores, nan=0.0)

def get_optimal_threshold(y_true, y_pred_proba, max_gap=0.25):
    """
    Optimise for F2 where recall > precision, and gap between recall and precision is <= max_gap.
    Fall back to F1.5 if no such threshold exists.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    gap = recall[:-1] - precision[:-1]
    valid_mask = (gap > 0) & (gap <= max_gap)

    if valid_mask.any():
        scores = calc_fbeta(precision, recall, beta=2)
        scores[~valid_mask] = -1
        optimal_idx = np.argmax(scores)
    else:
        print(f"No threshold satisfies recall >= precision with gap <= {max_gap}. Falling back to F1.5.")
        scores = calc_fbeta(precision, recall, beta=1.5)
        optimal_idx = np.argmax(scores)

    return thresholds[optimal_idx]

def get_constrained_threshold(y_true, y_pred_proba, prior_threshold, search_radius=0.15, max_gap=0.25):
    """
    For fine-tuning.
    Optimises threshold within a constrained range around a prior threshold.
    Falls back to prior if nothing satisfactory found.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # constrain search to within search_radius of the prior
    lower = prior_threshold - search_radius
    upper = prior_threshold + search_radius
    mask = (thresholds >= lower) & (thresholds <= upper)
    
    if not mask.any():
        return prior_threshold
    
    p = precision[:-1][mask]
    r = recall[:-1][mask]
    t = thresholds[mask]
    
    # apply gap constraint
    gap = r - p
    valid_mask = (gap > 0) & (gap <= max_gap)
    if valid_mask.any():
        scores = calc_fbeta(precision=p[valid_mask], recall=r[valid_mask], beta=2)
        return t[valid_mask][np.argmax(scores)]
    
    return prior_threshold

def get_evaluation_metrics(predictions):
    """
    Calculate metrics from predictions dict.
    """
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    y_pred_proba = predictions["y_pred_proba"]
    
    unique_labels = np.unique(y_true)
    has_both_classes = len(unique_labels) > 1
    
    if has_both_classes:
        # normal case: both classes are present
        auroc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
    else:
        # single class case
        if unique_labels[0] == 0:
            # 0% AF - specificity test
            auroc = None
            auprc = None
            precision = None
            recall = None
            f1 = None
            
            tn = np.sum(y_pred == 0)  # correct predictions
            fp = np.sum(y_pred == 1)  # false alarms
            specificity = tn / (tn + fp)
            
        else:
            # 100% AF - recall test
            auroc = None
            auprc = None
            specificity = None
            
            tp = np.sum(y_pred == 1)  # correct predictions
            fn = np.sum(y_pred == 0)  # missed detections
            recall = tp / (tp + fn)
            precision = 1.0 if tp > 0 else 0.0
            f1 = recall if tp > 0 else 0.0
        
    accuracy = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred_proba)
    prevalence = np.mean(y_true)
    brier_ref = prevalence * (1 - prevalence)
    bss = 1 - (brier / brier_ref) if brier_ref > 0 else None
    ece = expected_calibration_error(y_true, y_pred_proba)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "specificity": specificity,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "bss": bss,
        "ece": ece
    }

def get_median_threshold(horizon, input_window, folds_path):
    """
    Get median threshold from list of thresholds.
    """
    device = get_device()
    matches = glob.glob(
        f"{folds_path}/hor{horizon}_inp{input_window}_tar0.5_*"
    )
    if not matches:
        raise FileNotFoundError(
            f"No folder found for hor{horizon}_inp{input_window}_tar0.5"
        )
    hor_folder = matches[0]
    thresholds = []
    for fold in range(1, 6):
        model_path = f"{hor_folder}/fold{fold}/model.pt"
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
        thresholds.append(checkpoint["threshold"])
    median_threshold = float(np.median(thresholds))
    return median_threshold 

@torch.no_grad()
def get_predictions(model, dataloader, device, threshold=0.5):
    """
    Get predictions from trained model.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        device: Device to run evaluation on
        threshold: Classification threshold, default is 0.5
    
    Returns:
        dict: Contains y_true, y_pred, and y_pred_proba arrays
    """
    model.eval()
    y_true = []
    y_pred_proba = []
    
    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        probabilities = torch.sigmoid(logits)
        y_true.extend(y.cpu().numpy())
        y_pred_proba.extend(probabilities.cpu().detach().numpy())
    
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }

def get_warmup_lr(epoch, base_lr, warmup_epochs, start_factor=WARMUP_START_FACTOR):
    """
    Calculate learning rate with linear warmup.
    
    Args:
        epoch: Current epoch (0-indexed)
        base_lr: Target learning rate after warmup
        warmup_epochs: Number of epochs to warm up over
        start_factor: Starting LR as fraction of base_lr
    
    Returns:
        Learning rate for this epoch
    """
    if epoch < warmup_epochs:
        # linear warmup from start_factor * base_lr to base_lr
        alpha = epoch / warmup_epochs
        return base_lr * (start_factor + (1 - start_factor) * alpha)
    return base_lr

def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_pretrained_weights(model, pretrained_path, device):
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pretrained weights from {pretrained_path}")
    return model

def set_cnn_frozen(model, frozen=True):
    """
    Toggle requires_grad on the CNN backbone (block1..block9) only.

    The head — fc1, fc2, and the recurrent layer (lstm or gru, if present)
    — is always left trainable (requires_grad=True), regardless of `frozen`.

    frozen=True  -> backbone frozen   -> only the head trains (used in stage 1)
    frozen=False -> backbone unfrozen -> entire network trains (used in stage 2)
    """
    head_prefixes = ("fc1", "fc2", "lstm", "gru")

    matched = False
    for name, param in model.named_parameters():
        if name.startswith(head_prefixes):
            param.requires_grad = True
        else:
            param.requires_grad = not frozen
            matched = True

    if not matched:
        raise RuntimeError(
            "set_cnn_frozen found no CNN-backbone parameters to freeze — "
            "check model submodule naming (expected 'block1'..'block9') "
            "hasn't changed."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"{'Frozen' if frozen else 'Unfrozen'} CNN backbone — "
        f"trainable params: {trainable:,} / {total:,}"
    )

