import numpy as np


class LogisticRegressionScratch:
    def __init__(
        self,
        learning_rate=0.03,
        epochs=250,
        batch_size=512,
        l2=0.001,
        class_weight="balanced",
        tolerance=1e-6,
        patience=20,
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.class_weight = class_weight
        self.tolerance = tolerance
        self.patience = patience
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_weights(self, y):
        if self.class_weight != "balanced":
            return np.ones(len(y), dtype=float)
        n_samples = len(y)
        n_negative = max(int((y == 0).sum()), 1)
        n_positive = max(int((y == 1).sum()), 1)
        weight_negative = n_samples / (2.0 * n_negative)
        weight_positive = n_samples / (2.0 * n_positive)
        return np.where(y == 1, weight_positive, weight_negative)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        sample_weights = self._sample_weights(y)

        m_w = np.zeros(n_features)
        v_w = np.zeros(n_features)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        self.loss_history_ = []
        best_loss = np.inf
        stale_epochs = 0

        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]
                weight_batch = sample_weights[idx]

                probability = self._sigmoid(X_batch @ self.coef_ + self.intercept_)
                weighted_error = (probability - y_batch) * weight_batch

                grad_w = X_batch.T @ weighted_error / len(idx) + self.l2 * self.coef_
                grad_b = weighted_error.mean()

                step += 1
                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * (grad_w**2)
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b**2)

                m_w_hat = m_w / (1 - beta1**step)
                v_w_hat = v_w / (1 - beta2**step)
                m_b_hat = m_b / (1 - beta1**step)
                v_b_hat = v_b / (1 - beta2**step)

                self.coef_ -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.intercept_ -= (
                    self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
                )

            probability_full = self._sigmoid(X @ self.coef_ + self.intercept_)
            log_loss = -np.mean(
                sample_weights
                * (
                    y * np.log(probability_full + 1e-12)
                    + (1 - y) * np.log(1 - probability_full + 1e-12)
                )
            )
            loss = log_loss + 0.5 * self.l2 * np.dot(self.coef_, self.coef_)
            self.loss_history_.append(float(loss))

            if best_loss - loss > self.tolerance:
                best_loss = loss
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.patience:
                break

        self.n_iter_ = len(self.loss_history_)
        return self

    def predict_proba(self, X):
        positive_probability = self._sigmoid(
            np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        )
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
