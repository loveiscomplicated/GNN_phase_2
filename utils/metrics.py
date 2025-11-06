import numpy as np

def compute_metrics(y_true, y_pred, num_classes: int):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    acc = (yt == yp).mean() if yt.size else 0.0

    if num_classes == 2:
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return float(acc), float(precision), float(recall), float(f1)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f)
    precision = float(np.mean(precisions))
    recall    = float(np.mean(recalls))
    f1        = float(np.mean(f1s))
    return float(acc), precision, recall, f1