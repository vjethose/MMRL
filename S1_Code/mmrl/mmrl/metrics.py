import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize


def classification_metrics(probs, labels, num_classes):
    pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    out = {"accuracy": 100 * accuracy_score(labels, pred), "precision": 100 * p, "recall": 100 * r, "f1": 100 * f1,
           "confusion": confusion_matrix(labels, pred, labels=list(range(num_classes))).tolist()}
    try:
        y = label_binarize(labels, classes=list(range(num_classes)))
        present = y.sum(0) > 0
        out["map"] = 100 * float(average_precision_score(y[:, present], probs[:, present], average="macro")) if num_classes > 2 else float("nan")
        if num_classes == 2:
            out["auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            # macro one-vs-rest AUC over the classes present in the test set (a sampled test set can miss rare classes)
            out["auc"] = roc_auc_score(y[:, present], probs[:, present], average="macro")
            out["auc_classes"] = int(present.sum())
            # micro-averaged ROC curve, for the ROC figure
            fpr, tpr, _ = roc_curve(y.ravel(), probs.ravel())
            idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
            out["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    except ValueError:
        out["auc"] = float("nan")
    return out


def vqa_metrics(probs, labels):
    # answers are normalised when the dataset is built, so exact match = index match
    return {"accuracy": 100 * float((probs.argmax(1) == labels).mean()), "em": 100 * float((probs.argmax(1) == labels).mean())}


def retrieval_metrics(scores, relevant, ks=(1, 5, 10, 20, 50, 100)):
    """scores [Q, Ndoc]; relevant: list of sets of doc indices.
    AP@K = (1 / min(R, K)) * sum_{k<=K} P(k) rel(k)."""
    order = np.argsort(-scores, axis=1)
    out = {}
    for K in ks:
        aps, recs = [], []
        for q in range(scores.shape[0]):
            rel = relevant[q]
            if not rel:
                continue
            top = order[q, :K]
            hits = np.array([d in rel for d in top], dtype=np.float32)
            prec = np.cumsum(hits) / (np.arange(len(top)) + 1)
            aps.append(float((prec * hits).sum() / min(len(rel), K)))
            recs.append(float(hits.sum() / len(rel)))
        out[f"map@{K}"] = 100 * float(np.mean(aps)) if aps else float("nan")
        out[f"recall@{K}"] = 100 * float(np.mean(recs)) if recs else float("nan")
    out["accuracy"] = out.get("map@10", float("nan"))  # headline number for Table IV
    return out
