import numpy as np


class LinearSVMScratch:
    def __init__(
        self,
        learning_rate=0.01,
        epochs=250,
        batch_size=512,
        regularization=0.001,
        class_weight="balanced",
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.class_weight = class_weight
        self.random_state = random_state

    def _sample_weights(self, y_signed):
        if self.class_weight != "balanced":
            return np.ones(len(y_signed), dtype=float)
        n_samples = len(y_signed)
        n_negative = max(int((y_signed == -1).sum()), 1)
        n_positive = max(int((y_signed == 1).sum()), 1)
        weight_negative = n_samples / (2.0 * n_negative)
        weight_positive = n_samples / (2.0 * n_positive)
        return np.where(y_signed == 1, weight_positive, weight_negative)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_signed = np.where(np.asarray(y) == 1, 1.0, -1.0)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        sample_weights = self._sample_weights(y_signed)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            current_lr = self.learning_rate / (1.0 + 0.02 * epoch)

            for start in range(0, n_samples, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X[idx]
                y_batch = y_signed[idx]
                weight_batch = sample_weights[idx]

                margins = y_batch * (X_batch @ self.coef_ + self.intercept_)
                active = margins < 1.0

                grad_w = self.regularization * self.coef_
                grad_b = 0.0

                if np.any(active):
                    active_coefficient = weight_batch[active] * y_batch[active]
                    grad_w -= X_batch[active].T @ active_coefficient / len(idx)
                    grad_b -= active_coefficient.sum() / len(idx)

                self.coef_ -= current_lr * grad_w
                self.intercept_ -= current_lr * grad_b

            full_margin = y_signed * (X @ self.coef_ + self.intercept_)
            hinge = np.maximum(0.0, 1.0 - full_margin)
            loss = 0.5 * self.regularization * np.dot(self.coef_, self.coef_) + np.mean(
                sample_weights * hinge
            )
            self.loss_history_.append(float(loss))

        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(int)
