import numpy as np
import pandas as pd


class ScratchTabularPreprocessor:
    def fit(self, frame):
        self.numeric_cols_ = [
            c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
        ]
        self.categorical_cols_ = [
            c for c in frame.columns if c not in self.numeric_cols_
        ]

        self.medians_ = {
            c: float(frame[c].median()) for c in self.numeric_cols_
        }
        numeric_matrix = np.column_stack([
            frame[c].fillna(self.medians_[c]).astype(float).to_numpy()
            for c in self.numeric_cols_
        ])
        self.means_ = numeric_matrix.mean(axis=0)
        self.stds_ = numeric_matrix.std(axis=0)
        self.stds_[self.stds_ < 1e-12] = 1.0

        self.categories_ = {
            c: sorted(frame[c].fillna("__MISSING__").astype(str).unique().tolist())
            for c in self.categorical_cols_
        }

        self.feature_names_ = list(self.numeric_cols_)
        for c in self.categorical_cols_:
            self.feature_names_.extend([f"{c}={v}" for v in self.categories_[c]])
        return self

    def transform(self, frame):
        numeric_matrix = np.column_stack([
            frame[c].fillna(self.medians_[c]).astype(float).to_numpy()
            for c in self.numeric_cols_
        ])
        numeric_matrix = (numeric_matrix - self.means_) / self.stds_

        blocks = [numeric_matrix]
        for c in self.categorical_cols_:
            values = frame[c].fillna("__MISSING__").astype(str).to_numpy()
            one_hot = np.column_stack([
                (values == category).astype(float)
                for category in self.categories_[c]
            ])
            blocks.append(one_hot)

        return np.column_stack(blocks).astype(np.float64)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


class ScratchNonlinearPreprocessor:
    def __init__(self, n_quantiles=15):
        self.n_quantiles = n_quantiles

    def fit(self, frame):
        self.numeric_cols_ = [
            c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
        ]
        self.categorical_cols_ = [
            c for c in frame.columns if c not in self.numeric_cols_
        ]

        self.medians_ = {
            c: float(frame[c].median()) for c in self.numeric_cols_
        }
        numeric = np.column_stack([
            frame[c].fillna(self.medians_[c]).to_numpy(dtype=float)
            for c in self.numeric_cols_
        ])
        self.means_ = numeric.mean(axis=0)
        self.stds_ = numeric.std(axis=0)
        self.stds_[self.stds_ < 1e-12] = 1.0

        self.categories_ = {
            c: sorted(frame[c].fillna("__MISSING__").astype(str).unique().tolist())
            for c in self.categorical_cols_
        }

        candidate_keys = [
            "loan_percent_income",
            "loan_int_rate",
            "person_income",
            "credit_score",
            "loan_amnt",
        ]
        self.threshold_cols_ = [c for c in candidate_keys if c in frame.columns]
        quantile_grid = np.linspace(0.05, 0.95, self.n_quantiles)
        self.quantile_thresholds_ = {}
        for c in self.threshold_cols_:
            values = frame[c].fillna(self.medians_[c]).to_numpy(dtype=float)
            self.quantile_thresholds_[c] = np.unique(
                np.quantile(values, quantile_grid)
            )

        self.feature_names_ = list(self.numeric_cols_)
        for c in self.categorical_cols_:
            self.feature_names_.extend([
                f"{c}={category}" for category in self.categories_[c]
            ])
        for c in self.threshold_cols_:
            self.feature_names_.extend([
                f"{c}>q{i:02d}" for i in range(len(self.quantile_thresholds_[c]))
            ])

        if (
            "person_home_ownership" in self.categories_
            and "previous_loan_defaults_on_file" in self.categories_
        ):
            self.feature_names_.extend([
                f"home={h}|prev_default={d}"
                for h in self.categories_["person_home_ownership"]
                for d in self.categories_["previous_loan_defaults_on_file"]
            ])

        for c in ["loan_percent_income", "loan_int_rate", "person_income"]:
            if c in self.quantile_thresholds_ and "person_home_ownership" in self.categories_:
                self.feature_names_.extend([
                    f"{c}>q{i:02d}|home={h}"
                    for i in range(len(self.quantile_thresholds_[c]))
                    for h in self.categories_["person_home_ownership"]
                ])
        return self

    def transform(self, frame):
        numeric = np.column_stack([
            frame[c].fillna(self.medians_[c]).to_numpy(dtype=float)
            for c in self.numeric_cols_
        ])
        blocks = [(numeric - self.means_) / self.stds_]

        categorical_blocks = {}
        for c in self.categorical_cols_:
            values = frame[c].fillna("__MISSING__").astype(str).to_numpy()
            block = np.column_stack([
                (values == category).astype(float)
                for category in self.categories_[c]
            ])
            categorical_blocks[c] = block
            blocks.append(block)

        threshold_blocks = {}
        for c in self.threshold_cols_:
            values = frame[c].fillna(self.medians_[c]).to_numpy(dtype=float)
            block = np.column_stack([
                (values > threshold).astype(float)
                for threshold in self.quantile_thresholds_[c]
            ])
            threshold_blocks[c] = block
            blocks.append(block)

        if (
            "person_home_ownership" in categorical_blocks
            and "previous_loan_defaults_on_file" in categorical_blocks
        ):
            home = categorical_blocks["person_home_ownership"]
            previous = categorical_blocks["previous_loan_defaults_on_file"]
            blocks.append(
                (home[:, :, None] * previous[:, None, :]).reshape(len(frame), -1)
            )

        if "person_home_ownership" in categorical_blocks:
            home = categorical_blocks["person_home_ownership"]
            for c in ["loan_percent_income", "loan_int_rate", "person_income"]:
                if c in threshold_blocks:
                    threshold_block = threshold_blocks[c]
                    blocks.append(
                        (threshold_block[:, :, None] * home[:, None, :])
                        .reshape(len(frame), -1)
                    )

        return np.column_stack(blocks).astype(np.float64)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)