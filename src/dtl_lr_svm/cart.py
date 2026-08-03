import numpy as np


class CARTClassifierScratch:
    def __init__(
        self,
        max_depth=7,
        min_samples_split=30,
        min_samples_leaf=15,
        max_thresholds=None,
        max_features=None,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.max_features = max_features
        self.random_state = random_state

    @staticmethod
    def _gini_from_positive_count(positive_count, n_samples):
        if n_samples <= 0:
            return 0.0
        p = positive_count / n_samples
        return 2.0 * p * (1.0 - p)

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        self.n_features_in_ = self.X_.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        self.tree_ = self._grow(np.arange(len(self.y_)), depth=0)

        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def _make_leaf(self, indices):
        probability = float(self.y_[indices].mean()) if len(indices) else 0.0
        return {
            "is_leaf": True,
            "probability": probability,
            "prediction": int(probability >= 0.5),
            "n_samples": int(len(indices)),
        }

    def _get_feature_subset(self):
        if self.max_features is None or self.max_features >= self.n_features_in_:
            return np.arange(self.n_features_in_)
        if self.max_features == "sqrt":
            n_features = max(1, int(np.sqrt(self.n_features_in_)))
        else:
            n_features = int(self.max_features)
        return self.rng_.choice(self.n_features_in_, size=n_features, replace=False)

    def _best_split(self, indices):
        y_node = self.y_[indices]
        n_node = len(indices)
        parent_gini = self._gini_from_positive_count(y_node.sum(), n_node)
        best_gain = 0.0
        best_split = None

        for feature_index in self._get_feature_subset():
            values = self.X_[indices, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_y = y_node[order]

            candidate_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            candidate_positions = candidate_positions[
                (candidate_positions + 1 >= self.min_samples_leaf)
                & (n_node - (candidate_positions + 1) >= self.min_samples_leaf)
            ]
            if len(candidate_positions) == 0:
                continue

            if (
                self.max_thresholds is not None
                and len(candidate_positions) > self.max_thresholds
            ):
                selected = np.linspace(
                    0, len(candidate_positions) - 1, self.max_thresholds, dtype=int
                )
                candidate_positions = candidate_positions[selected]

            cumulative_positive = np.cumsum(sorted_y)
            n_left = candidate_positions + 1
            n_right = n_node - n_left
            positive_left = cumulative_positive[candidate_positions]
            positive_right = cumulative_positive[-1] - positive_left

            p_left = positive_left / n_left
            p_right = positive_right / n_right
            gini_left = 2.0 * p_left * (1.0 - p_left)
            gini_right = 2.0 * p_right * (1.0 - p_right)

            gains = parent_gini - (
                (n_left / n_node) * gini_left + (n_right / n_node) * gini_right
            )

            local_best = int(np.argmax(gains))
            gain = float(gains[local_best])
            if gain <= best_gain + 1e-12:
                continue

            split_position = int(candidate_positions[local_best])
            threshold = float(
                (sorted_values[split_position] + sorted_values[split_position + 1])
                / 2.0
            )
            left_mask = values <= threshold

            best_gain = gain
            best_split = (
                feature_index,
                threshold,
                indices[left_mask],
                indices[~left_mask],
                gain,
            )

        return best_split

    def _grow(self, indices, depth):
        stopping_condition = (
            depth >= self.max_depth
            or len(indices) < self.min_samples_split
            or np.unique(self.y_[indices]).size == 1
        )
        if stopping_condition:
            return self._make_leaf(indices)

        split = self._best_split(indices)
        if split is None:
            return self._make_leaf(indices)

        feature_index, threshold, left_indices, right_indices, gain = split
        self.feature_importances_[feature_index] += gain * len(indices)

        return {
            "is_leaf": False,
            "feature_index": int(feature_index),
            "threshold": float(threshold),
            "n_samples": int(len(indices)),
            "left": self._grow(left_indices, depth + 1),
            "right": self._grow(right_indices, depth + 1),
        }

    def _predict_probability_one(self, row):
        node = self.tree_
        while not node["is_leaf"]:
            if row[node["feature_index"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["probability"]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        positive_probability = np.fromiter(
            (self._predict_probability_one(row) for row in X),
            dtype=float,
            count=len(X),
        )
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
