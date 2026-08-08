import numpy as np


def stratified_train_validation_split(y, validation_size=0.20, random_state=42):
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    train_indices, validation_indices = [], []

    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        n_validation = int(round(len(cls_indices) * validation_size))
        validation_indices.extend(cls_indices[:n_validation])
        train_indices.extend(cls_indices[n_validation:])

    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def stratified_kfold_indices(y, n_splits=5, random_state=42):
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    fold_validation = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        for fold_id, part in enumerate(np.array_split(cls_indices, n_splits)):
            fold_validation[fold_id].extend(part.tolist())

    all_indices = np.arange(len(y))
    for fold_id in range(n_splits):
        valid_indices = np.asarray(fold_validation[fold_id], dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[valid_indices] = False
        yield all_indices[train_mask], valid_indices


def macro_f1_numpy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    f1_pos = 2.0 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2.0 * tn / max(2 * tn + fp + fn, 1)
    return 0.5 * (f1_pos + f1_neg)


def optimize_macro_f1_threshold(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    sorted_score = score[order]
    tp = np.r_[0, np.cumsum(y_sorted == 1)]
    fp = np.r_[0, np.cumsum(y_sorted == 0)]
    total_pos = int((y_true == 1).sum())
    total_neg = int((y_true == 0).sum())
    fn = total_pos - tp
    tn = total_neg - fp
    f1_pos = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    f1_neg = 2.0 * tn / np.maximum(2.0 * tn + fp + fn, 1.0)
    macro = 0.5 * (f1_pos + f1_neg)

    valid_k = [0]
    if len(score) > 1:
        valid_k.extend(
            (np.flatnonzero(sorted_score[:-1] > sorted_score[1:]) + 1).tolist()
        )
    valid_k.append(len(score))
    valid_k = np.asarray(sorted(set(valid_k)), dtype=int)
    best_k = int(valid_k[np.argmax(macro[valid_k])])
    if best_k == 0:
        threshold = float(sorted_score[0] + 1e-12)
    elif best_k == len(score):
        threshold = float(sorted_score[-1] - 1e-12)
    else:
        threshold = float((sorted_score[best_k - 1] + sorted_score[best_k]) / 2.0)
    return threshold, float(macro[best_k])


__all__ = [
    "stratified_train_validation_split",
    "stratified_kfold_indices",
    "macro_f1_numpy",
    "optimize_macro_f1_threshold",
]
