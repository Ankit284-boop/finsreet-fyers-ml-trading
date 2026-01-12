import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def walk_forward_predict_proba(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    min_train: int
) -> pd.DataFrame:
    df = df.copy()
    df["prob_up"] = np.nan

    for i in range(min_train, len(df) - 1):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
        ])

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1][0]
        df.loc[i, "prob_up"] = float(np.clip(prob, 0.05, 0.95))

    return df
