"""
AdaBoost pipeline for the credit default prediction project.

Expected inputs in the same folder:
- train_processed.parquet  -> contains TARGET and SK_ID_CURR
- test_processed.parquet   -> contains SK_ID_CURR and the same model features as train

What this script does:
1) Loads processed parquet files.
2) Creates a leakage-safe train/validation split.
3) Tunes a small AdaBoost grid with stratified CV using ROC-AUC.
4) Evaluates on a holdout validation set with ROC-AUC, PR-AUC, Precision, Recall, F1.
5) Selects a threshold that maximizes F1 on the validation set.
6) Retrains the best pipeline on the full training set.
7) Writes predictions for the provided test set.
8) Saves metrics, threshold table, feature importances, and plots.

Notes:
- Your processed data are already numeric/encoded, so AdaBoost only needs imputation.
- Because the target is imbalanced (~8% defaults), the script uses balanced sample weights.
- Base learner is a shallow decision tree, which is the standard weak learner for AdaBoost.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_processed.parquet"
TEST_PATH = BASE_DIR / "test_processed.parquet"
OUTPUT_DIR = BASE_DIR / "adaboost_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    # Stumps (max_depth=1) are classical for AdaBoost, but depth=2 can help on tabular data.
    base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                AdaBoostClassifier(
                    estimator=base_tree,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns one fewer threshold than precision/recall points.
    threshold_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    threshold_df["f1"] = (
        2
        * threshold_df["precision"]
        * threshold_df["recall"]
        / (threshold_df["precision"] + threshold_df["recall"] + 1e-12)
    )

    best_idx = int(threshold_df["f1"].idxmax())
    best_threshold = float(np.asarray(threshold_df["threshold"].to_numpy()[best_idx]).item())
    return best_threshold, threshold_df.sort_values("f1", ascending=False)

def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AdaBoost (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, save_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AdaBoost (PR-AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_feature_importance(feature_names: list[str], importances: np.ndarray, save_path: Path, top_n: int = 20) -> pd.DataFrame:
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    top_fi = fi.head(top_n).sort_values("importance")

    plt.figure(figsize=(8, 7))
    plt.barh(top_fi["feature"], top_fi["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} AdaBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return fi



def main() -> None:
    print("Loading processed data...")
    train_df = load_parquet(TRAIN_PATH)
    test_df = load_parquet(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Default rate: {train_df[TARGET_COL].mean():.4%}")

    X, y = prepare_xy(train_df)
    X_test = test_df.copy()

    # Keep ID for submission/prediction export, but do not use it as a feature.
    test_ids = X_test[ID_COL].copy() if ID_COL in X_test.columns else pd.Series(np.arange(len(X_test)), name=ID_COL)
    train_ids = X[ID_COL].copy() if ID_COL in X.columns else None

    drop_cols = [c for c in [ID_COL] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)

    # Safety check: align test columns to training features.
    X_test = X_test.reindex(columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
        "model__estimator__max_depth": [1, 2],
        "model__estimator__min_samples_leaf": [1, 20, 100],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Running GridSearchCV for AdaBoost...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train, model__sample_weight=train_weights)

    best_model = grid.best_estimator_
    print("Best parameters:")
    print(grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.6f}")

    val_prob = best_model.predict_proba(X_val)[:, 1]

    best_threshold, threshold_table = find_best_f1_threshold(y_val, val_prob)

    metrics_default_05 = evaluate_predictions(y_val, val_prob, threshold=0.50)
    metrics_best_f1 = evaluate_predictions(y_val, val_prob, threshold=best_threshold)

    report_best_f1 = classification_report(
        y_val,
        (val_prob >= best_threshold).astype(int),
        zero_division=0,
        output_dict=True,
    )

    metrics_payload = {
        "best_params": grid.best_params_,
        "best_cv_roc_auc": float(grid.best_score_),
        "validation_metrics_threshold_0.5": metrics_default_05,
        "validation_metrics_best_f1_threshold": metrics_best_f1,
        "best_f1_threshold": float(best_threshold),
        "classification_report_best_f1_threshold": report_best_f1,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    threshold_table.to_csv(OUTPUT_DIR / "validation_threshold_search.csv", index=False)
    plot_roc_curve(y_val, val_prob, OUTPUT_DIR / "roc_curve.png")
    plot_pr_curve(y_val, val_prob, OUTPUT_DIR / "pr_curve.png")

    # Retrain the best model on the full training set before generating test predictions.
    print("Retraining best AdaBoost pipeline on full training data...")
    final_model = clone(best_model)
    full_weights = compute_sample_weight(class_weight="balanced", y=y)
    final_model.fit(X, y, model__sample_weight=full_weights)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    pred_df = pd.DataFrame(
        {
            ID_COL: test_ids,
            "adaboost_default_probability": test_prob,
            "adaboost_predicted_label": test_pred,
        }
    )
    pred_df.to_csv(OUTPUT_DIR / "adaboost_test_predictions.csv", index=False)

    # Feature importances come from the fitted AdaBoost model.
    model_step = final_model.named_steps["model"]
    feature_importances = getattr(model_step, "feature_importances_", None)
    if feature_importances is not None:
        fi_df = plot_feature_importance(
            feature_names=X.columns.tolist(),
            importances=feature_importances,
            save_path=OUTPUT_DIR / "feature_importance.png",
            top_n=20,
        )
        fi_df.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)

    print("Done.")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
