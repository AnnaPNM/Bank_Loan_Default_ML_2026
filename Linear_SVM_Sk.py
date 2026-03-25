"""
Master reproducible automated runner for two experiment families:
1) Adaptive feature-selection Linear SVM methods:
   - none
   - filter_f_classif
   - embedded_l1
2) Fixed EDA-ranked feature subset experiments:
   - top-1 through top-20 ranked features
"""

import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "optuna": "optuna",
    "sklearn": "scikit-learn",
    "psutil": "psutil",
    "GPUtil": "GPUtil",
}


def ensure_packages_installed(packages: dict) -> None:
    missing = []
    for module_name, pip_name in packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(pip_name)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages_installed(REQUIRED_PACKAGES)

import json
import random
import warnings
import time
import threading
import shutil
import re
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, SelectFromModel, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import psutil
except Exception:
    psutil = None

try:
    import GPUtil
except Exception:
    GPUtil = None


# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_CANDIDATES = ["SK_ID_CURR", "ID", "id"]

TRAIN_PATH = Path("train_processed.csv")
TEST_PATH = Path("test_processed.csv")

OUT_ROOT = Path("svm_master_automated_outputs")
ADAPTIVE_OUT_ROOT = OUT_ROOT / "adaptive_feature_selection_methods"
TOPK_OUT_ROOT = OUT_ROOT / "topk_eda_feature_runs"

VALID_SIZE = 0.20
LOW_CARDINALITY_THRESHOLD = 20

FEATURE_SELECTION_METHODS = ["none", "filter_f_classif", "embedded_l1"]
FILTER_PERCENTILE_OPTIONS = [10, 20, 30, 40, 50, 60, 80]
EMBEDDED_SELECTOR_C_MIN = 1e-4
EMBEDDED_SELECTOR_C_MAX = 1e1
EMBEDDED_SELECTOR_CLASS_WEIGHT = "balanced"
EMBEDDED_SELECTOR_MAX_ITER = 5000

EDA_RANKED_FEATURES = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "DAYS_BIRTH",
    "REGION_RATING_CLIENT_W_CITY",
    "REGION_RATING_CLIENT",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_ID_PUBLISH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_DAY",
]
MAX_TOPK = 20

OPTUNA_TRIALS = 12
OPTUNA_TIMEOUT = None
INNER_SCORING = "average_precision"
N_INNER_SPLITS = 3

CALIBRATION_CV = 3
N_OOF_THRESHOLD_SPLITS = 3
THRESHOLD_GRID_SIZE = 300

SAVE_BINARY_TEST_PREDICTIONS = True
TOP_N_COEFFICIENTS = 25
DIAG_SAMPLE_INTERVAL_SEC = 1.0


# ============================================================
# HELPERS
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def find_id_column(df: pd.DataFrame):
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    return None



def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "cp1252", "latin1", "iso-8859-1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not read {path}. Last error: {last_error}")



def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)



def make_calibrated_classifier(estimator, method="sigmoid", cv=3):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)



def is_binary_like(series: pd.Series) -> bool:
    vals = sorted(pd.Series(series).dropna().unique().tolist())
    return set(vals).issubset({0, 1, 0.0, 1.0, 0.5})



def looks_continuous_by_name(col: str) -> bool:
    c = col.upper()
    continuous_tokens = [
        "AMT_", "DAYS_", "EXT_SOURCE", "REGION_", "OBS_", "DEF_",
        "CREDIT_INCOME", "ANNUITY_INCOME", "CREDIT_ANNUITY",
        "EMPLOYED_BIRTH", "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_PROD",
    ]
    return any(tok in c for tok in continuous_tokens)



def infer_feature_types(X: pd.DataFrame):
    categorical_by_dtype = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    categorical_by_name = []
    categorical_tokens = [
        "NAME_", "CODE_", "TYPE", "MODE", "WEEKDAY", "ORGANIZATION",
        "FONDKAPREMONT", "HOUSETYPE", "WALLSMATERIAL", "EMERGENCYSTATE",
        "OCCUPATION_TYPE",
    ]
    for col in X.columns:
        up = col.upper()
        if any(tok in up for tok in categorical_tokens):
            categorical_by_name.append(col)

    categorical_by_low_card = []
    for col in X.columns:
        s = X[col]
        nunique = s.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(s):
            if is_binary_like(s):
                continue
            if looks_continuous_by_name(col):
                continue
            if nunique <= LOW_CARDINALITY_THRESHOLD:
                categorical_by_low_card.append(col)

    categorical_cols = sorted(set(categorical_by_dtype + categorical_by_name + categorical_by_low_card))
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols



def build_preprocessor(X: pd.DataFrame):
    numeric_cols, categorical_cols = infer_feature_types(X)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, numeric_cols, categorical_cols



def build_selector(selector_method: str):
    if selector_method == "none":
        return "passthrough"
    if selector_method == "filter_f_classif":
        return SelectPercentile(score_func=f_classif, percentile=50)
    if selector_method == "embedded_l1":
        selector_estimator = LinearSVC(
            penalty="l1",
            dual=False,
            C=0.01,
            class_weight=EMBEDDED_SELECTOR_CLASS_WEIGHT,
            max_iter=EMBEDDED_SELECTOR_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        return SelectFromModel(estimator=selector_estimator)
    raise ValueError(f"Unsupported selector_method: {selector_method}")



def build_adaptive_pipeline(preprocessor, selector_method: str):
    selector = build_selector(selector_method)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("svm", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=20000,
                random_state=RANDOM_STATE,
            )),
        ]
    )



def build_fixed_pipeline(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svm", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=20000,
                random_state=RANDOM_STATE,
            )),
        ]
    )



def score_from_decision(y_true, decision_scores, metric: str):
    if metric == "average_precision":
        return average_precision_score(y_true, decision_scores)
    if metric == "roc_auc":
        return roc_auc_score(y_true, decision_scores)
    raise ValueError(f"Unsupported metric: {metric}")



def metrics_at_threshold(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }, y_pred


def confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }



def threshold_search_table(y_true, y_score, n_thresholds=300):
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)



def choose_threshold_from_oof(y_true, y_score, n_thresholds=300):
    table = threshold_search_table(y_true, y_score, n_thresholds=n_thresholds)
    return float(table.loc[0, "threshold"]), table



def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)



def save_confusion_matrix(y_true, y_pred, title, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def save_curves(y_true, y_score, out_dir: Path, prefix: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_score):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower().replace(' ', '_')}_roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_score):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower().replace(' ', '_')}_pr_curve.png", dpi=200)
    plt.close()



def get_preprocessed_feature_names(fitted_preprocessor):
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        return None



def inspect_selected_features(fitted_pipeline):
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(fitted_preprocessor)
    selector = fitted_pipeline.named_steps["selector"]

    if selector == "passthrough":
        if feature_names is None:
            return pd.DataFrame({"feature_name": [], "selected": []}), {
                "n_preprocessed_features": None,
                "n_selected_features": None,
                "selection_rate": None,
                "selector_type": "passthrough",
            }
        selected_df = pd.DataFrame({"feature_name": feature_names, "selected": True})
        return selected_df, {
            "n_preprocessed_features": int(len(feature_names)),
            "n_selected_features": int(len(feature_names)),
            "selection_rate": 1.0,
            "selector_type": "passthrough",
        }

    if feature_names is None:
        return pd.DataFrame({"feature_name": [], "selected": []}), {
            "n_preprocessed_features": None,
            "n_selected_features": None,
            "selection_rate": None,
            "selector_type": type(selector).__name__,
        }

    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        selected_df = pd.DataFrame({"feature_name": feature_names, "selected": mask.astype(bool)})
        n_total = int(len(feature_names))
        n_selected = int(mask.sum())
        return selected_df, {
            "n_preprocessed_features": n_total,
            "n_selected_features": n_selected,
            "selection_rate": float(n_selected / n_total) if n_total else None,
            "selector_type": type(selector).__name__,
        }

    return pd.DataFrame({"feature_name": feature_names, "selected": np.nan}), {
        "n_preprocessed_features": int(len(feature_names)),
        "n_selected_features": None,
        "selection_rate": None,
        "selector_type": type(selector).__name__,
    }



def infer_source_feature_name(processed_feature_name, numeric_cols, categorical_cols):
    name = str(processed_feature_name)
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        remainder = name.replace("cat__", "", 1)
        for col in sorted(categorical_cols, key=len, reverse=True):
            prefix = f"{col}_"
            if remainder == col or remainder.startswith(prefix):
                return col
        return remainder
    return name



def map_feature_family(source_col, categorical_cols):
    col = str(source_col)
    up = col.upper()
    if col in categorical_cols:
        return "categorical"
    if "EXT_SOURCE" in up:
        return "external_scores"
    if up.startswith("AMT_"):
        return "amounts"
    if up.startswith("DAYS_"):
        return "days"
    if up.startswith("CNT_"):
        return "counts"
    if up.startswith("FLAG_"):
        return "flags"
    if any(tok in up for tok in [
        "APARTMENTS", "BASEMENTAREA", "COMMONAREA", "ELEVATORS", "ENTRANCES",
        "FLOORSMAX", "FLOORSMIN", "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA",
        "NONLIVINGAPARTMENTS", "NONLIVINGAREA", "YEARS_BEGINEXPLUATATION",
        "YEARS_BUILD", "HOUSETYPE", "WALLSMATERIAL", "EMERGENCYSTATE",
    ]):
        return "housing"
    if "REGION" in up:
        return "regional"
    if "PHONE" in up or "MOBIL" in up or "EMAIL" in up:
        return "contact"
    if "NAME_" in up or "CODE_" in up or "ORGANIZATION" in up or "OCCUPATION" in up:
        return "categorical"
    return "other"



def extract_svm_coefficients_adaptive(fitted_pipeline, numeric_cols, categorical_cols):
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(fitted_preprocessor)
    if feature_names is None:
        raise ValueError("Could not extract preprocessed feature names from fitted preprocessor.")

    selector = fitted_pipeline.named_steps["selector"]
    svm = fitted_pipeline.named_steps["svm"]
    if not hasattr(svm, "coef_"):
        raise ValueError("Final SVM model does not expose coef_.")
    coef = svm.coef_.ravel()

    if selector == "passthrough":
        selected_feature_names = np.array(feature_names)
    else:
        if not hasattr(selector, "get_support"):
            raise ValueError("Selector does not expose get_support().")
        selected_feature_names = np.array(feature_names)[selector.get_support()]

    if len(selected_feature_names) != len(coef):
        raise ValueError(
            f"Feature/coefficient length mismatch: {len(selected_feature_names)} names vs {len(coef)} coefficients."
        )

    coef_df = pd.DataFrame({
        "processed_feature_name": selected_feature_names,
        "source_feature": [infer_source_feature_name(name, numeric_cols, categorical_cols) for name in selected_feature_names],
        "coefficient": coef,
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["direction"] = np.where(coef_df["coefficient"] >= 0, "positive_to_class_1", "negative_to_class_1")
    coef_df["feature_family"] = coef_df["source_feature"].apply(lambda x: map_feature_family(x, categorical_cols))
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)



def extract_svm_coefficients_fixed(fitted_pipeline, numeric_cols, categorical_cols):
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(fitted_preprocessor)
    if feature_names is None:
        raise ValueError("Could not extract preprocessed feature names from fitted preprocessor.")

    svm = fitted_pipeline.named_steps["svm"]
    if not hasattr(svm, "coef_"):
        raise ValueError("Final SVM model does not expose coef_.")
    coef = svm.coef_.ravel()

    if len(feature_names) != len(coef):
        raise ValueError(f"Feature/coefficient length mismatch: {len(feature_names)} names vs {len(coef)} coefficients.")

    coef_df = pd.DataFrame({
        "processed_feature_name": feature_names,
        "source_feature": [infer_source_feature_name(name, numeric_cols, categorical_cols) for name in feature_names],
        "coefficient": coef,
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["direction"] = np.where(coef_df["coefficient"] >= 0, "positive_to_class_1", "negative_to_class_1")
    coef_df["feature_family"] = coef_df["source_feature"].apply(lambda x: map_feature_family(x, categorical_cols))
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)



def plot_top_coefficients(coef_df, out_dir, top_n=25):
    top_abs = coef_df.head(top_n).copy().sort_values("abs_coefficient", ascending=True)
    plt.figure(figsize=(10, max(6, top_n * 0.25)))
    plt.barh(top_abs["processed_feature_name"], top_abs["abs_coefficient"])
    plt.xlabel("Absolute coefficient magnitude")
    plt.title(f"Top {top_n} absolute Linear SVM coefficients")
    plt.tight_layout()
    plt.savefig(out_dir / "top_absolute_coefficients.png", dpi=200)
    plt.close()

    top_pos = coef_df[coef_df["coefficient"] > 0].head(top_n).copy().sort_values("coefficient", ascending=True)
    if not top_pos.empty:
        plt.figure(figsize=(10, max(6, len(top_pos) * 0.25)))
        plt.barh(top_pos["processed_feature_name"], top_pos["coefficient"])
        plt.xlabel("Coefficient")
        plt.title(f"Top {min(top_n, len(top_pos))} positive Linear SVM coefficients")
        plt.tight_layout()
        plt.savefig(out_dir / "top_positive_coefficients.png", dpi=200)
        plt.close()

    top_neg = coef_df[coef_df["coefficient"] < 0].head(top_n).copy().sort_values("coefficient", ascending=False)
    if not top_neg.empty:
        plt.figure(figsize=(10, max(6, len(top_neg) * 0.25)))
        plt.barh(top_neg["processed_feature_name"], top_neg["coefficient"])
        plt.xlabel("Coefficient")
        plt.title(f"Top {min(top_n, len(top_neg))} negative Linear SVM coefficients")
        plt.tight_layout()
        plt.savefig(out_dir / "top_negative_coefficients.png", dpi=200)
        plt.close()



def summarize_feature_families(coef_df):
    grouped = coef_df.groupby("feature_family").agg(
        n_selected_processed_features=("processed_feature_name", "count"),
        n_unique_source_features=("source_feature", "nunique"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        max_abs_coefficient=("abs_coefficient", "max"),
    ).reset_index()
    return grouped.sort_values(["max_abs_coefficient", "n_selected_processed_features"], ascending=[False, False])



def sanitize_name_component(text: str, max_len: int = 40) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "na"



def make_feature_set_dirname(k: int, selected_features: list, preview_count: int = 4) -> str:
    selected_features = list(selected_features)
    preview = "__".join(sanitize_name_component(f) for f in selected_features[:preview_count])
    full_signature = "||".join(map(str, selected_features))
    short_hash = hashlib.md5(full_signature.encode("utf-8")).hexdigest()[:10]
    return f"top_{k}_features__{preview}__{short_hash}"



def optuna_objective_factory_adaptive(X_train, y_train, base_pipeline, scoring_metric, n_inner_splits, random_state, selector_method):
    def objective(trial):
        params = {}
        if selector_method == "filter_f_classif":
            params["selector__percentile"] = trial.suggest_categorical("selector__percentile", FILTER_PERCENTILE_OPTIONS)
        elif selector_method == "embedded_l1":
            params["selector__estimator__C"] = trial.suggest_float(
                "selector__estimator__C", EMBEDDED_SELECTOR_C_MIN, EMBEDDED_SELECTOR_C_MAX, log=True
            )

        params["svm__C"] = trial.suggest_float("svm__C", 1e-3, 1e2, log=True)
        params["svm__loss"] = trial.suggest_categorical("svm__loss", ["hinge", "squared_hinge"])
        class_weight_choice = trial.suggest_categorical("svm__class_weight", ["none", "balanced"])
        params["svm__class_weight"] = None if class_weight_choice == "none" else "balanced"

        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        fold_scores = []
        for tr_idx, va_idx in inner_cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            model = clone(base_pipeline).set_params(**params)
            model.fit(X_tr, y_tr)
            decision_scores = model.decision_function(X_va)
            fold_scores.append(score_from_decision(y_va, decision_scores, scoring_metric))
        return float(np.mean(fold_scores))

    return objective



def tune_with_optuna_adaptive(X_train, y_train, base_pipeline, scoring_metric, n_inner_splits, n_trials, timeout, random_state, selector_method):
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = optuna_objective_factory_adaptive(
        X_train=X_train,
        y_train=y_train,
        base_pipeline=base_pipeline,
        scoring_metric=scoring_metric,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
        selector_method=selector_method,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    if "svm__class_weight" in best_params and best_params["svm__class_weight"] == "none":
        best_params["svm__class_weight"] = None
    return study, best_params, float(study.best_value)



def optuna_objective_factory_fixed(X_train, y_train, base_pipeline, scoring_metric, n_inner_splits, random_state):
    def objective(trial):
        params = {
            "svm__C": trial.suggest_float("svm__C", 1e-3, 1e2, log=True),
            "svm__loss": trial.suggest_categorical("svm__loss", ["hinge", "squared_hinge"]),
        }
        class_weight_choice = trial.suggest_categorical("svm__class_weight", ["none", "balanced"])
        params["svm__class_weight"] = None if class_weight_choice == "none" else "balanced"

        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        fold_scores = []
        for tr_idx, va_idx in inner_cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            model = clone(base_pipeline).set_params(**params)
            model.fit(X_tr, y_tr)
            decision_scores = model.decision_function(X_va)
            fold_scores.append(score_from_decision(y_va, decision_scores, scoring_metric))
        return float(np.mean(fold_scores))

    return objective



def tune_with_optuna_fixed(X_train, y_train, base_pipeline, scoring_metric, n_inner_splits, n_trials, timeout, random_state):
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = optuna_objective_factory_fixed(
        X_train=X_train,
        y_train=y_train,
        base_pipeline=base_pipeline,
        scoring_metric=scoring_metric,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    if "svm__class_weight" in best_params and best_params["svm__class_weight"] == "none":
        best_params["svm__class_weight"] = None
    return study, best_params, float(study.best_value)



def generate_oof_calibrated_probabilities(X, y, base_pipeline, best_params, outer_folds=3, calib_folds=3, random_state=42):
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        fold_estimator = clone(base_pipeline).set_params(**best_params)
        calibrated_fold_model = make_calibrated_classifier(estimator=fold_estimator, method="sigmoid", cv=calib_folds)
        calibrated_fold_model.fit(X_tr, y_tr)
        oof_proba[va_idx] = calibrated_fold_model.predict_proba(X_va)[:, 1]
        print(f"  OOF fold {fold}/{outer_folds} done.")
    return oof_proba


# ============================================================
# RUNTIME / RESOURCE DIAGNOSTICS
# ============================================================
def safe_mean(series):
    series = pd.Series(series).dropna()
    return float(series.mean()) if len(series) else np.nan



def safe_max(series):
    series = pd.Series(series).dropna()
    return float(series.max()) if len(series) else np.nan



def safe_last(series):
    series = pd.Series(series).dropna()
    return float(series.iloc[-1]) if len(series) else np.nan



def format_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def get_gpu_snapshot():
    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                mem_total = float(gpu.memoryTotal)
                mem_used = float(gpu.memoryUsed)
                mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else np.nan
                return {
                    "gpu_available": True,
                    "gpu_name": gpu.name,
                    "gpu_util_percent": float(gpu.load * 100.0),
                    "gpu_mem_used_mb": mem_used,
                    "gpu_mem_total_mb": mem_total,
                    "gpu_mem_percent": mem_pct,
                }
        except Exception:
            pass

    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    name, util, mem_used, mem_total = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                    mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else np.nan
                    return {
                        "gpu_available": True,
                        "gpu_name": name,
                        "gpu_util_percent": util,
                        "gpu_mem_used_mb": mem_used,
                        "gpu_mem_total_mb": mem_total,
                        "gpu_mem_percent": mem_pct,
                    }
        except Exception:
            pass

    return {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_util_percent": np.nan,
        "gpu_mem_used_mb": np.nan,
        "gpu_mem_total_mb": np.nan,
        "gpu_mem_percent": np.nan,
    }


class RunDiagnosticsMonitor:
    def __init__(self, out_dir: Path, interval_sec: float = 1.0):
        self.out_dir = out_dir
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []
        self.start_time_wall = None
        self.end_time_wall = None
        self.process = None
        self.cpu_count = (psutil.cpu_count(logical=True) if psutil else 1) or 1

    def start(self):
        self.start_time_wall = time.time()
        if psutil:
            self.process = psutil.Process()
            psutil.cpu_percent(interval=None)
            self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _get_process_peak_memory_mb(self):
        if not psutil or self.process is None:
            return np.nan
        try:
            mem_info = self.process.memory_info()
            if hasattr(mem_info, "peak_wset"):
                return float(mem_info.peak_wset) / (1024 ** 2)
        except Exception:
            pass
        return np.nan

    def _sample_once(self):
        now = time.time()
        elapsed_sec = now - self.start_time_wall if self.start_time_wall else np.nan

        if psutil and self.process is not None:
            vm = psutil.virtual_memory()
            process_mem = self.process.memory_info().rss / (1024 ** 2)
            process_peak_mem = self._get_process_peak_memory_mb()
            system_cpu_pct = psutil.cpu_percent(interval=None)
            process_cpu_pct_total = self.process.cpu_percent(interval=None)
            process_cpu_pct_normalized = process_cpu_pct_total / self.cpu_count
            ram_used_gb = float(vm.used) / (1024 ** 3)
            ram_total_gb = float(vm.total) / (1024 ** 3)
            ram_percent = float(vm.percent)
        else:
            process_mem = process_peak_mem = system_cpu_pct = process_cpu_pct_normalized = np.nan
            ram_used_gb = ram_total_gb = ram_percent = np.nan

        gpu_info = get_gpu_snapshot()

        self.samples.append({
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_sec,
            "elapsed_minutes": elapsed_sec / 60.0 if pd.notna(elapsed_sec) else np.nan,
            "elapsed_hms": format_hms(elapsed_sec) if pd.notna(elapsed_sec) else None,
            "system_cpu_percent": float(system_cpu_pct) if pd.notna(system_cpu_pct) else np.nan,
            "process_cpu_percent": float(process_cpu_pct_normalized) if pd.notna(process_cpu_pct_normalized) else np.nan,
            "process_memory_rss_mb": float(process_mem) if pd.notna(process_mem) else np.nan,
            "process_memory_peak_mb": float(process_peak_mem) if pd.notna(process_peak_mem) else np.nan,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "ram_percent": ram_percent,
            "gpu_available": gpu_info["gpu_available"],
            "gpu_name": gpu_info["gpu_name"],
            "gpu_util_percent": gpu_info["gpu_util_percent"],
            "gpu_mem_used_mb": gpu_info["gpu_mem_used_mb"],
            "gpu_mem_total_mb": gpu_info["gpu_mem_total_mb"],
            "gpu_mem_percent": gpu_info["gpu_mem_percent"],
        })

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def stop_and_save(self, extra_summary: dict | None = None):
        self.end_time_wall = time.time()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        if not self.samples:
            try:
                self._sample_once()
            except Exception:
                pass

        df = pd.DataFrame(self.samples)
        if df.empty:
            runtime_sec = float(self.end_time_wall - self.start_time_wall) if self.start_time_wall else np.nan
            df = pd.DataFrame([{
                "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": runtime_sec,
                "elapsed_minutes": runtime_sec / 60.0 if pd.notna(runtime_sec) else np.nan,
                "elapsed_hms": format_hms(runtime_sec) if pd.notna(runtime_sec) else None,
            }])

        df.to_csv(self.out_dir / "run_diagnostics_samples.csv", index=False)

        runtime_sec = float(self.end_time_wall - self.start_time_wall) if self.start_time_wall else np.nan
        runtime_min = runtime_sec / 60.0 if pd.notna(runtime_sec) else np.nan

        summary = {
            "start_time_iso": datetime.fromtimestamp(self.start_time_wall).isoformat(timespec="seconds") if self.start_time_wall else None,
            "end_time_iso": datetime.fromtimestamp(self.end_time_wall).isoformat(timespec="seconds") if self.end_time_wall else None,
            "runtime_seconds": runtime_sec,
            "runtime_minutes": runtime_min,
            "runtime_hms": format_hms(runtime_sec) if pd.notna(runtime_sec) else None,
            "system_cpu_percent_avg": safe_mean(df.get("system_cpu_percent", [])),
            "system_cpu_percent_max": safe_max(df.get("system_cpu_percent", [])),
            "process_cpu_percent_avg": safe_mean(df.get("process_cpu_percent", [])),
            "process_cpu_percent_max": safe_max(df.get("process_cpu_percent", [])),
            "process_memory_rss_mb_avg": safe_mean(df.get("process_memory_rss_mb", [])),
            "process_memory_rss_mb_max": safe_max(df.get("process_memory_rss_mb", [])),
            "process_memory_rss_mb_last": safe_last(df.get("process_memory_rss_mb", [])),
            "process_memory_peak_mb_max": safe_max(df.get("process_memory_peak_mb", [])),
            "ram_used_gb_avg": safe_mean(df.get("ram_used_gb", [])),
            "ram_used_gb_max": safe_max(df.get("ram_used_gb", [])),
            "ram_used_gb_last": safe_last(df.get("ram_used_gb", [])),
            "ram_percent_avg": safe_mean(df.get("ram_percent", [])),
            "ram_percent_max": safe_max(df.get("ram_percent", [])),
            "ram_percent_last": safe_last(df.get("ram_percent", [])),
            "gpu_available_any": bool(pd.Series(df.get("gpu_available", [])).fillna(False).any()) if "gpu_available" in df.columns else False,
            "gpu_name": df["gpu_name"].dropna().iloc[0] if ("gpu_name" in df.columns and not df["gpu_name"].dropna().empty) else None,
            "gpu_util_percent_avg": safe_mean(df.get("gpu_util_percent", [])),
            "gpu_util_percent_max": safe_max(df.get("gpu_util_percent", [])),
            "gpu_mem_used_mb_avg": safe_mean(df.get("gpu_mem_used_mb", [])),
            "gpu_mem_used_mb_max": safe_max(df.get("gpu_mem_used_mb", [])),
            "gpu_mem_used_mb_last": safe_last(df.get("gpu_mem_used_mb", [])),
            "gpu_mem_percent_avg": safe_mean(df.get("gpu_mem_percent", [])),
            "gpu_mem_percent_max": safe_max(df.get("gpu_mem_percent", [])),
        }
        if extra_summary:
            summary.update(extra_summary)

        pd.DataFrame([summary]).to_csv(self.out_dir / "run_diagnostics_summary.csv", index=False)
        save_json(summary, self.out_dir / "run_diagnostics_summary.json")


# ============================================================
# DATA LOADING
# ============================================================
def load_and_prepare_data():
    print("Reading CSV files...")
    train_df = safe_read_csv(TRAIN_PATH)
    test_df = safe_read_csv(TEST_PATH)

    print(f"train shape: {train_df.shape}")
    print(f"test shape : {test_df.shape}")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"{TARGET_COL} not found in training CSV.")

    id_col = find_id_column(train_df)
    test_id_col = find_id_column(test_df)

    y = train_df[TARGET_COL].astype(int).copy()
    drop_cols_train = [TARGET_COL] + ([id_col] if id_col else [])
    drop_cols_test = [test_id_col] if test_id_col else []

    X = train_df.drop(columns=drop_cols_train, errors="ignore").copy()
    X_test = test_df.drop(columns=drop_cols_test, errors="ignore").copy()

    missing_in_test = [c for c in X.columns if c not in X_test.columns]
    extra_in_test = [c for c in X_test.columns if c not in X.columns]

    if missing_in_test:
        for c in missing_in_test:
            X_test[c] = np.nan

    if extra_in_test:
        X_test = X_test.drop(columns=extra_in_test)

    X_test = X_test[X.columns]

    print(f"Model train matrix shape: {X.shape}")
    print(f"Model test matrix shape : {X_test.shape}")
    print(f"Positive class rate     : {y.mean():.6f}")

    return test_df, X, y, X_test, test_id_col


# ============================================================
# ADAPTIVE METHODS RUNNER
# ============================================================
def run_single_adaptive_method(feature_selection_method: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, test_df: pd.DataFrame, test_id_col):
    method_tag = str(feature_selection_method).strip().lower()
    out_dir = ADAPTIVE_OUT_ROOT / f"svm_light_fs_interpret_outputs_{method_tag}"
    ensure_dir(out_dir)

    diagnostics = RunDiagnosticsMonitor(out_dir=out_dir, interval_sec=DIAG_SAMPLE_INTERVAL_SEC)
    diagnostics.start()

    run_status = "success"
    run_error_message = ""

    try:
        print("\n" + "=" * 70)
        print(f"RUNNING ADAPTIVE METHOD: {feature_selection_method}")
        print("=" * 70)

        preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
        print(f"Numeric columns used     : {len(numeric_cols)}")
        print(f"Categorical columns used : {len(categorical_cols)}")

        base_pipeline = build_adaptive_pipeline(preprocessor, selector_method=feature_selection_method)
        print(f"Feature selection method : {feature_selection_method}")

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=VALID_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        print(f"Train split shape: {X_train.shape}")
        print(f"Valid split shape: {X_valid.shape}")
        print(f"Train positive rate: {y_train.mean():.6f}")
        print(f"Valid positive rate: {y_valid.mean():.6f}")

        print("Running Optuna tuning on training split...")
        study, best_params, best_inner_score = tune_with_optuna_adaptive(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            base_pipeline=base_pipeline,
            scoring_metric=INNER_SCORING,
            n_inner_splits=N_INNER_SPLITS,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            random_state=RANDOM_STATE,
            selector_method=feature_selection_method,
        )

        print("Best params:", best_params)
        print(f"Best inner {INNER_SCORING}: {best_inner_score:.6f}")

        print("Fitting inspection model for selected-feature and coefficient analysis...")
        selector_inspection_model = clone(base_pipeline).set_params(**best_params)
        selector_inspection_model.fit(X_train, y_train)

        selected_features_df, selected_meta = inspect_selected_features(selector_inspection_model)
        pd.DataFrame([selected_meta]).to_csv(out_dir / "selected_feature_summary.csv", index=False)
        selected_features_df.to_csv(out_dir / "selected_feature_mask.csv", index=False)
        selected_features_df[selected_features_df["selected"] == True].copy().to_csv(out_dir / "selected_feature_names.csv", index=False)

        coef_df = extract_svm_coefficients_adaptive(
            selector_inspection_model,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        coef_df.to_csv(out_dir / "svm_all_selected_coefficients.csv", index=False)
        coef_df.head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_absolute_coefficients.csv", index=False)
        coef_df[coef_df["coefficient"] > 0].head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_positive_coefficients.csv", index=False)
        coef_df[coef_df["coefficient"] < 0].head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_negative_coefficients.csv", index=False)

        family_summary_df = summarize_feature_families(coef_df)
        family_summary_df.to_csv(out_dir / "svm_coefficient_family_summary.csv", index=False)
        plot_top_coefficients(coef_df, out_dir, top_n=TOP_N_COEFFICIENTS)

        print("Generating OOF-calibrated probabilities for threshold tuning...")
        oof_proba_train = generate_oof_calibrated_probabilities(
            X=X_train.reset_index(drop=True),
            y=y_train.reset_index(drop=True),
            base_pipeline=base_pipeline,
            best_params=best_params,
            outer_folds=N_OOF_THRESHOLD_SPLITS,
            calib_folds=CALIBRATION_CV,
            random_state=RANDOM_STATE,
        )

        best_threshold, threshold_table = choose_threshold_from_oof(
            y_true=y_train.reset_index(drop=True),
            y_score=oof_proba_train,
            n_thresholds=THRESHOLD_GRID_SIZE,
        )
        print(f"Best OOF threshold by F1: {best_threshold:.6f}")
        threshold_table.to_csv(out_dir / "oof_threshold_search_table.csv", index=False)

        final_model = make_calibrated_classifier(
            estimator=clone(base_pipeline).set_params(**best_params),
            method="sigmoid",
            cv=CALIBRATION_CV,
        )
        final_model.fit(X_train, y_train)

        valid_proba = final_model.predict_proba(X_valid)[:, 1]
        default_metrics, y_pred_default = metrics_at_threshold(y_valid, valid_proba, threshold=0.50)
        tuned_metrics, y_pred_tuned = metrics_at_threshold(y_valid, valid_proba, threshold=best_threshold)
        default_counts = confusion_counts(y_valid, y_pred_default)
        tuned_counts = confusion_counts(y_valid, y_pred_tuned)

        pd.DataFrame(classification_report(y_valid, y_pred_default, output_dict=True)).transpose().to_csv(
            out_dir / "classification_report_threshold_0_50.csv"
        )
        pd.DataFrame(classification_report(y_valid, y_pred_tuned, output_dict=True)).transpose().to_csv(
            out_dir / "classification_report_oof_best_threshold.csv"
        )

        save_confusion_matrix(y_valid, y_pred_default, "Validation Confusion Matrix (threshold=0.50)", out_dir / "confusion_matrix_threshold_0_50.png")
        save_confusion_matrix(y_valid, y_pred_tuned, f"Validation Confusion Matrix (threshold={best_threshold:.4f})", out_dir / "confusion_matrix_oof_best_threshold.png")
        save_curves(y_valid, valid_proba, out_dir, prefix=f"Validation {feature_selection_method}")

        test_proba = final_model.predict_proba(X_test)[:, 1]
        if test_id_col is not None:
            submission = pd.DataFrame({test_id_col: test_df[test_id_col].values, TARGET_COL: test_proba})
        else:
            submission = pd.DataFrame({TARGET_COL: test_proba})
        submission.to_csv(out_dir / "svm_submission_probabilities.csv", index=False)
        submission.to_csv(out_dir / "svm_test_probabilities.csv", index=False)

        if SAVE_BINARY_TEST_PREDICTIONS:
            test_pred_binary = (test_proba >= best_threshold).astype(int)
            if test_id_col is not None:
                binary_submission = pd.DataFrame({test_id_col: test_df[test_id_col].values, TARGET_COL: test_pred_binary})
            else:
                binary_submission = pd.DataFrame({TARGET_COL: test_pred_binary})
            binary_submission.to_csv(out_dir / "svm_test_predictions_binary_best_threshold.csv", index=False)

        save_json({
            "feature_selection_method": feature_selection_method,
            "best_params": best_params,
            "best_inner_score": float(best_inner_score),
            "inner_scoring_metric": INNER_SCORING,
            "oof_best_threshold": float(best_threshold),
        }, out_dir / "final_model_settings.json")

        pd.DataFrame([{
            "feature_selection_method": feature_selection_method,
            "best_inner_score": float(best_inner_score),
            "best_threshold": float(best_threshold),
            "default_threshold": 0.50,
            "default_accuracy": float(default_metrics["accuracy"]),
            "default_precision": float(default_metrics["precision"]),
            "default_recall": float(default_metrics["recall"]),
            "default_f1": float(default_metrics["f1"]),
            "default_roc_auc": float(default_metrics["roc_auc"]),
            "default_pr_auc": float(default_metrics["pr_auc"]),
            "default_tp": int(default_counts["tp"]),
            "default_tn": int(default_counts["tn"]),
            "default_fp": int(default_counts["fp"]),
            "default_fn": int(default_counts["fn"]),
            "tuned_accuracy": float(tuned_metrics["accuracy"]),
            "tuned_precision": float(tuned_metrics["precision"]),
            "tuned_recall": float(tuned_metrics["recall"]),
            "tuned_f1": float(tuned_metrics["f1"]),
            "tuned_roc_auc": float(tuned_metrics["roc_auc"]),
            "tuned_pr_auc": float(tuned_metrics["pr_auc"]),
            "tuned_tp": int(tuned_counts["tp"]),
            "tuned_tn": int(tuned_counts["tn"]),
            "tuned_fp": int(tuned_counts["fp"]),
            "tuned_fn": int(tuned_counts["fn"]),
            "n_preprocessed_features": selected_meta.get("n_preprocessed_features"),
            "n_selected_features": selected_meta.get("n_selected_features"),
            "selection_rate": selected_meta.get("selection_rate"),
        }]).to_csv(out_dir / "method_summary.csv", index=False)

        return {
            "experiment_family": "adaptive_feature_selection",
            "feature_selection_method": feature_selection_method,
            "out_dir": str(out_dir),
            "best_inner_score": float(best_inner_score),
            "best_threshold": float(best_threshold),
            "default_threshold": 0.50,
            "default_accuracy": float(default_metrics["accuracy"]),
            "default_precision": float(default_metrics["precision"]),
            "default_recall": float(default_metrics["recall"]),
            "default_f1": float(default_metrics["f1"]),
            "default_roc_auc": float(default_metrics["roc_auc"]),
            "default_pr_auc": float(default_metrics["pr_auc"]),
            "default_tp": int(default_counts["tp"]),
            "default_tn": int(default_counts["tn"]),
            "default_fp": int(default_counts["fp"]),
            "default_fn": int(default_counts["fn"]),
            "tuned_accuracy": float(tuned_metrics["accuracy"]),
            "tuned_precision": float(tuned_metrics["precision"]),
            "tuned_recall": float(tuned_metrics["recall"]),
            "tuned_f1": float(tuned_metrics["f1"]),
            "tuned_roc_auc": float(tuned_metrics["roc_auc"]),
            "tuned_pr_auc": float(tuned_metrics["pr_auc"]),
            "tuned_tp": int(tuned_counts["tp"]),
            "tuned_tn": int(tuned_counts["tn"]),
            "tuned_fp": int(tuned_counts["fp"]),
            "tuned_fn": int(tuned_counts["fn"]),
            "n_preprocessed_features": selected_meta.get("n_preprocessed_features"),
            "n_selected_features": selected_meta.get("n_selected_features"),
            "selection_rate": selected_meta.get("selection_rate"),
            "status": "success",
        }

    except Exception as e:
        run_status = "failed"
        run_error_message = str(e)
        print(f"\nAdaptive method {feature_selection_method} failed: {e}")
        return {
            "experiment_family": "adaptive_feature_selection",
            "feature_selection_method": feature_selection_method,
            "out_dir": str(out_dir),
            "status": "failed",
            "error_message": str(e),
        }

    finally:
        diagnostics.stop_and_save(extra_summary={
            "feature_selection_method": feature_selection_method,
            "status": run_status,
            "error_message": run_error_message,
        })


# ============================================================
# TOP-K FIXED EDA RUNNER
# ============================================================
def run_topk_experiment(k: int, available_eda_features: list, X_full: pd.DataFrame, y_full: pd.Series, X_test_full: pd.DataFrame, test_df: pd.DataFrame, test_id_col):
    selected_features = available_eda_features[:k]
    dir_name = make_feature_set_dirname(k=k, selected_features=selected_features, preview_count=4)
    out_dir = TOPK_OUT_ROOT / dir_name
    ensure_dir(out_dir)

    diagnostics = RunDiagnosticsMonitor(out_dir=out_dir, interval_sec=DIAG_SAMPLE_INTERVAL_SEC)
    diagnostics.start()
    run_status = "success"
    run_error_message = ""

    try:
        X = X_full[selected_features].copy()
        X_test = X_test_full[selected_features].copy()
        y = y_full.copy()

        pd.DataFrame({
            "selected_feature_rank": range(1, len(selected_features) + 1),
            "selected_feature_name": selected_features,
        }).to_csv(out_dir / "selected_feature_names_ordered.csv", index=False)

        print(f"\n===== RUN WITH TOP-{k} FEATURES =====")
        print("Selected features:", selected_features)

        preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
        base_pipeline = build_fixed_pipeline(preprocessor)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=VALID_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        print(f"Train split shape: {X_train.shape}")
        print(f"Valid split shape: {X_valid.shape}")

        study, best_params, best_inner_score = tune_with_optuna_fixed(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            base_pipeline=base_pipeline,
            scoring_metric=INNER_SCORING,
            n_inner_splits=N_INNER_SPLITS,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            random_state=RANDOM_STATE,
        )

        print("Best params:", best_params)
        print(f"Best inner {INNER_SCORING}: {best_inner_score:.6f}")

        inspection_model = clone(base_pipeline).set_params(**best_params)
        inspection_model.fit(X_train, y_train)

        coef_df = extract_svm_coefficients_fixed(
            inspection_model,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        coef_df.to_csv(out_dir / "svm_all_coefficients.csv", index=False)
        coef_df.head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_absolute_coefficients.csv", index=False)
        coef_df[coef_df["coefficient"] > 0].head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_positive_coefficients.csv", index=False)
        coef_df[coef_df["coefficient"] < 0].head(TOP_N_COEFFICIENTS).to_csv(out_dir / "svm_top_negative_coefficients.csv", index=False)

        family_summary_df = summarize_feature_families(coef_df)
        family_summary_df.to_csv(out_dir / "svm_coefficient_family_summary.csv", index=False)
        plot_top_coefficients(coef_df, out_dir, top_n=TOP_N_COEFFICIENTS)

        oof_proba_train = generate_oof_calibrated_probabilities(
            X=X_train.reset_index(drop=True),
            y=y_train.reset_index(drop=True),
            base_pipeline=base_pipeline,
            best_params=best_params,
            outer_folds=N_OOF_THRESHOLD_SPLITS,
            calib_folds=CALIBRATION_CV,
            random_state=RANDOM_STATE,
        )
        best_threshold, threshold_table = choose_threshold_from_oof(
            y_true=y_train.reset_index(drop=True),
            y_score=oof_proba_train,
            n_thresholds=THRESHOLD_GRID_SIZE,
        )
        threshold_table.to_csv(out_dir / "oof_threshold_search_table.csv", index=False)

        final_model = make_calibrated_classifier(
            estimator=clone(base_pipeline).set_params(**best_params),
            method="sigmoid",
            cv=CALIBRATION_CV,
        )
        final_model.fit(X_train, y_train)

        valid_proba = final_model.predict_proba(X_valid)[:, 1]
        default_metrics, y_pred_default = metrics_at_threshold(y_valid, valid_proba, threshold=0.50)
        tuned_metrics, y_pred_tuned = metrics_at_threshold(y_valid, valid_proba, threshold=best_threshold)
        default_counts = confusion_counts(y_valid, y_pred_default)
        tuned_counts = confusion_counts(y_valid, y_pred_tuned)

        pd.DataFrame(classification_report(y_valid, y_pred_default, output_dict=True)).transpose().to_csv(
            out_dir / "classification_report_threshold_0_50.csv"
        )
        pd.DataFrame(classification_report(y_valid, y_pred_tuned, output_dict=True)).transpose().to_csv(
            out_dir / "classification_report_oof_best_threshold.csv"
        )

        save_confusion_matrix(y_valid, y_pred_default, f"Validation Confusion Matrix (Top-{k}, threshold=0.50)", out_dir / "confusion_matrix_threshold_0_50.png")
        save_confusion_matrix(y_valid, y_pred_tuned, f"Validation Confusion Matrix (Top-{k}, threshold={best_threshold:.4f})", out_dir / "confusion_matrix_oof_best_threshold.png")
        save_curves(y_valid, valid_proba, out_dir, prefix=f"Validation Top-{k}")

        final_model.fit(X, y)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        if test_id_col is not None:
            submission = pd.DataFrame({test_id_col: test_df[test_id_col].values, TARGET_COL: test_proba})
        else:
            submission = pd.DataFrame({TARGET_COL: test_proba})
        submission.to_csv(out_dir / "svm_submission_probabilities.csv", index=False)
        submission.to_csv(out_dir / "svm_test_probabilities.csv", index=False)

        if SAVE_BINARY_TEST_PREDICTIONS:
            test_pred_binary = (test_proba >= best_threshold).astype(int)
            if test_id_col is not None:
                binary_submission = pd.DataFrame({test_id_col: test_df[test_id_col].values, TARGET_COL: test_pred_binary})
            else:
                binary_submission = pd.DataFrame({TARGET_COL: test_pred_binary})
            binary_submission.to_csv(out_dir / "svm_test_predictions_binary_best_threshold.csv", index=False)

        save_json({
            "top_k": k,
            "selected_features": selected_features,
            "best_params": best_params,
            "best_inner_score": float(best_inner_score),
            "inner_scoring_metric": INNER_SCORING,
            "oof_best_threshold": float(best_threshold),
        }, out_dir / "final_model_settings.json")

        return {
            "experiment_family": "fixed_topk_eda",
            "top_k": k,
            "out_dir": str(out_dir),
            "selected_features": selected_features,
            "n_selected_features": len(selected_features),
            "best_threshold": float(best_threshold),
            "best_inner_score": float(best_inner_score),
            "default_accuracy": float(default_metrics["accuracy"]),
            "default_precision": float(default_metrics["precision"]),
            "default_recall": float(default_metrics["recall"]),
            "default_f1": float(default_metrics["f1"]),
            "default_roc_auc": float(default_metrics["roc_auc"]),
            "default_pr_auc": float(default_metrics["pr_auc"]),
            "tuned_accuracy": float(tuned_metrics["accuracy"]),
            "tuned_precision": float(tuned_metrics["precision"]),
            "tuned_recall": float(tuned_metrics["recall"]),
            "tuned_f1": float(tuned_metrics["f1"]),
            "tuned_roc_auc": float(tuned_metrics["roc_auc"]),
            "tuned_pr_auc": float(tuned_metrics["pr_auc"]),
            "status": "success",
        }

    except Exception as e:
        run_status = "failed"
        run_error_message = str(e)
        print(f"\nTop-{k} run failed: {e}")
        return {
            "experiment_family": "fixed_topk_eda",
            "top_k": k,
            "out_dir": str(out_dir),
            "selected_features": selected_features,
            "n_selected_features": len(selected_features),
            "status": "failed",
            "error_message": str(e),
        }

    finally:
        diagnostics.stop_and_save(extra_summary={
            "top_k": k,
            "selected_features": selected_features,
            "status": run_status,
            "error_message": run_error_message,
        })


# ============================================================
# PLOTS / MASTER SUMMARIES
# ============================================================
def plot_adaptive_summary(summary_df: pd.DataFrame, out_root: Path):
    plot_df = summary_df.dropna(subset=["tuned_f1"]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["feature_selection_method"], plot_df["tuned_f1"])
    plt.ylabel("Tuned F1")
    plt.xlabel("Method")
    plt.title("Tuned F1 across adaptive SVM methods")
    plt.tight_layout()
    plt.savefig(out_root / "adaptive_methods_tuned_f1_comparison.png", dpi=200)
    plt.close()



def plot_topk_summary(results_df: pd.DataFrame, out_root: Path):
    plot_df = results_df.dropna(subset=["top_k", "tuned_f1"]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["top_k"], plot_df["default_f1"], marker="o", label="Default threshold F1")
    plt.plot(plot_df["top_k"], plot_df["tuned_f1"], marker="o", label="OOF-tuned threshold F1")
    best_idx = plot_df["tuned_f1"].idxmax()
    best_k = int(plot_df.loc[best_idx, "top_k"])
    best_f1 = float(plot_df.loc[best_idx, "tuned_f1"])
    plt.scatter([best_k], [best_f1], s=100, label=f"Best tuned F1: k={best_k}, F1={best_f1:.4f}")
    plt.xlabel("Number of selected features")
    plt.ylabel("F1 score")
    plt.title("F1 Score vs Number of Selected Features (Top-1 to Top-20)")
    plt.xticks(range(1, MAX_TOPK + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "f1_vs_number_of_selected_features.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["top_k"], plot_df["tuned_f1"], marker="o")
    plt.scatter([best_k], [best_f1], s=100)
    plt.xlabel("Number of selected features")
    plt.ylabel("Tuned F1")
    plt.title("Best tuned F1 across Top-k EDA feature runs")
    plt.xticks(range(1, MAX_TOPK + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_root / "tuned_f1_vs_number_of_selected_features.png", dpi=200)
    plt.close()



def make_requested_comparison_table(adaptive_df: pd.DataFrame, topk_df: pd.DataFrame) -> pd.DataFrame:
    method_name_map = {
        "none": "None (all features)",
        "filter_f_classif": "Filter F-classif",
        "embedded_l1": "Embedded L1",
    }

    rows = []

    if adaptive_df is not None and not adaptive_df.empty:
        for _, row in adaptive_df.iterrows():
            if str(row.get("status", "success")).lower() != "success":
                continue
            method_label = method_name_map.get(row.get("feature_selection_method"), str(row.get("feature_selection_method")))
            rows.append({
                "Method": method_label,
                "Setting": "Default",
                "Thr.": float(row.get("default_threshold", 0.500)),
                "Acc.": float(row.get("default_accuracy", np.nan)),
                "Precision": float(row.get("default_precision", np.nan)),
                "Recall": float(row.get("default_recall", np.nan)),
                "F1": float(row.get("default_f1", np.nan)),
                "ROC-AUC": float(row.get("default_roc_auc", np.nan)),
                "PR-AUC": float(row.get("default_pr_auc", np.nan)),
                "TP": int(row.get("default_tp", 0)),
                "TN": int(row.get("default_tn", 0)),
                "FP": int(row.get("default_fp", 0)),
                "FN": int(row.get("default_fn", 0)),
            })
            rows.append({
                "Method": method_label,
                "Setting": "Tuned",
                "Thr.": float(row.get("best_threshold", np.nan)),
                "Acc.": float(row.get("tuned_accuracy", np.nan)),
                "Precision": float(row.get("tuned_precision", np.nan)),
                "Recall": float(row.get("tuned_recall", np.nan)),
                "F1": float(row.get("tuned_f1", np.nan)),
                "ROC-AUC": float(row.get("tuned_roc_auc", np.nan)),
                "PR-AUC": float(row.get("tuned_pr_auc", np.nan)),
                "TP": int(row.get("tuned_tp", 0)),
                "TN": int(row.get("tuned_tn", 0)),
                "FP": int(row.get("tuned_fp", 0)),
                "FN": int(row.get("tuned_fn", 0)),
            })

    topk_success = pd.DataFrame()
    if topk_df is not None and not topk_df.empty:
        if "status" in topk_df.columns:
            topk_success = topk_df[topk_df["status"].astype(str).str.lower() == "success"].copy()
        else:
            topk_success = topk_df.copy()

    if not topk_success.empty and "tuned_f1" in topk_success.columns:
        fixed_row = topk_success.loc[topk_success["tuned_f1"].idxmax()].copy()
        rows.append({
            "Method": "Fixed EDA features",
            "Setting": "Default",
            "Thr.": float(fixed_row.get("default_threshold", 0.500)),
            "Acc.": float(fixed_row.get("default_accuracy", np.nan)),
            "Precision": float(fixed_row.get("default_precision", np.nan)),
            "Recall": float(fixed_row.get("default_recall", np.nan)),
            "F1": float(fixed_row.get("default_f1", np.nan)),
            "ROC-AUC": float(fixed_row.get("default_roc_auc", np.nan)),
            "PR-AUC": float(fixed_row.get("default_pr_auc", np.nan)),
            "TP": int(fixed_row.get("default_tp", 0)),
            "TN": int(fixed_row.get("default_tn", 0)),
            "FP": int(fixed_row.get("default_fp", 0)),
            "FN": int(fixed_row.get("default_fn", 0)),
        })
        rows.append({
            "Method": "Fixed EDA features",
            "Setting": "Tuned",
            "Thr.": float(fixed_row.get("best_threshold", np.nan)),
            "Acc.": float(fixed_row.get("tuned_accuracy", np.nan)),
            "Precision": float(fixed_row.get("tuned_precision", np.nan)),
            "Recall": float(fixed_row.get("tuned_recall", np.nan)),
            "F1": float(fixed_row.get("tuned_f1", np.nan)),
            "ROC-AUC": float(fixed_row.get("tuned_roc_auc", np.nan)),
            "PR-AUC": float(fixed_row.get("tuned_pr_auc", np.nan)),
            "TP": int(fixed_row.get("tuned_tp", 0)),
            "TN": int(fixed_row.get("tuned_tn", 0)),
            "FP": int(fixed_row.get("tuned_fp", 0)),
            "FN": int(fixed_row.get("tuned_fn", 0)),
        })

    return pd.DataFrame(rows, columns=[
        "Method", "Setting", "Thr.", "Acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TP", "TN", "FP", "FN"
    ])


def build_master_summary(adaptive_df: pd.DataFrame, topk_df: pd.DataFrame, available_eda_features: list, missing_eda_features: list):
    adaptive_best = None
    if not adaptive_df.empty and "tuned_f1" in adaptive_df.columns:
        adaptive_candidates = adaptive_df.dropna(subset=["tuned_f1"]).copy()
        if not adaptive_candidates.empty:
            row = adaptive_candidates.loc[adaptive_candidates["tuned_f1"].idxmax()].to_dict()
            adaptive_best = row

    topk_best = None
    if not topk_df.empty and "tuned_f1" in topk_df.columns:
        topk_candidates = topk_df.dropna(subset=["tuned_f1"]).copy()
        if not topk_candidates.empty:
            row = topk_candidates.loc[topk_candidates["tuned_f1"].idxmax()].to_dict()
            topk_best = row

    overall_best = None
    candidates = []
    if adaptive_best is not None:
        candidates.append((adaptive_best.get("tuned_f1", np.nan), adaptive_best))
    if topk_best is not None:
        candidates.append((topk_best.get("tuned_f1", np.nan), topk_best))
    candidates = [c for c in candidates if pd.notna(c[0])]
    if candidates:
        overall_best = max(candidates, key=lambda x: x[0])[1]

    return {
        "output_root": str(OUT_ROOT.resolve()),
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "adaptive_methods_run": FEATURE_SELECTION_METHODS,
        "max_topk_run": MAX_TOPK,
        "available_eda_features": available_eda_features,
        "missing_eda_features": missing_eda_features,
        "best_adaptive_run": adaptive_best,
        "best_topk_run": topk_best,
        "overall_best_run": overall_best,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    seed_everything(RANDOM_STATE)
    ensure_dir(OUT_ROOT)
    ensure_dir(ADAPTIVE_OUT_ROOT)
    ensure_dir(TOPK_OUT_ROOT)

    test_df, X, y, X_test, test_id_col = load_and_prepare_data()

    available_eda_features = [c for c in EDA_RANKED_FEATURES if c in X.columns]
    missing_eda_features = [c for c in EDA_RANKED_FEATURES if c not in X.columns]

    print(f"Available EDA-ranked features: {len(available_eda_features)} / {len(EDA_RANKED_FEATURES)}")
    if missing_eda_features:
        print("Missing from dataset:", missing_eda_features)

    if len(available_eda_features) < MAX_TOPK:
        raise ValueError(f"Need at least {MAX_TOPK} available ranked features, but found {len(available_eda_features)}.")

    adaptive_summary_rows = []
    for feature_selection_method in FEATURE_SELECTION_METHODS:
        result = run_single_adaptive_method(
            feature_selection_method=feature_selection_method,
            X=X,
            y=y,
            X_test=X_test,
            test_df=test_df,
            test_id_col=test_id_col,
        )
        adaptive_summary_rows.append(result)

    adaptive_summary_df = pd.DataFrame(adaptive_summary_rows)
    adaptive_summary_df.to_csv(ADAPTIVE_OUT_ROOT / "all_methods_summary.csv", index=False)
    plot_adaptive_summary(adaptive_summary_df, ADAPTIVE_OUT_ROOT)

    topk_results = []
    for k in range(1, MAX_TOPK + 1):
        result = run_topk_experiment(
            k=k,
            available_eda_features=available_eda_features,
            X_full=X,
            y_full=y,
            X_test_full=X_test,
            test_df=test_df,
            test_id_col=test_id_col,
        )
        topk_results.append(result)

    topk_results_df = pd.DataFrame(topk_results).sort_values("top_k").reset_index(drop=True)
    topk_results_df.to_csv(TOPK_OUT_ROOT / "all_top1_to_top20_results_summary.csv", index=False)
    plot_topk_summary(topk_results_df, TOPK_OUT_ROOT)

    adaptive_master = adaptive_summary_df.copy()
    topk_master = topk_results_df.copy()

    adaptive_master.insert(0, "experiment_group", "adaptive_feature_selection")
    topk_master.insert(0, "experiment_group", "fixed_topk_eda")
    combined_summary_df = pd.concat([adaptive_master, topk_master], ignore_index=True, sort=False)
    combined_summary_df.to_csv(OUT_ROOT / "combined_experiments_summary.csv", index=False)

    requested_comparison_df = make_requested_comparison_table(
        adaptive_df=adaptive_summary_df,
        topk_df=topk_results_df,
    )
    requested_comparison_df.to_csv(OUT_ROOT / "all_methods_3plus1_comparison_table.csv", index=False)

    master_summary = build_master_summary(
        adaptive_df=adaptive_summary_df,
        topk_df=topk_results_df,
        available_eda_features=available_eda_features,
        missing_eda_features=missing_eda_features,
    )
    save_json(master_summary, OUT_ROOT / "master_summary.json")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 70)
    print(f"Adaptive summary saved to: {ADAPTIVE_OUT_ROOT / 'all_methods_summary.csv'}")
    print(f"Top-k summary saved to   : {TOPK_OUT_ROOT / 'all_top1_to_top20_results_summary.csv'}")
    print(f"Combined summary saved to: {OUT_ROOT / 'combined_experiments_summary.csv'}")
    print(f"3+1 comparison saved to : {OUT_ROOT / 'all_methods_3plus1_comparison_table.csv'}")
    print(f"Master JSON saved to     : {OUT_ROOT / 'master_summary.json'}")

    if not adaptive_summary_df.empty:
        print("\nAdaptive methods summary:")
        print(adaptive_summary_df.to_string(index=False))

    if not topk_results_df.empty:
        print("\nTop-k summary:")
        display_cols = [c for c in ["top_k", "tuned_f1", "default_f1", "best_threshold", "status"] if c in topk_results_df.columns]
        print(topk_results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
