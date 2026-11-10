"""
What this script does
  1. Loads data/cohort.csv (made by src/build_cohort.py).
  2. Uses the time-based split: start 2005-2015 = train, 2016-2019 = test.
  3. Builds features from 12 categorical + 10 numeric raw features:
       - categorical -> one-hot (missing kept as its own category)
       - intervention_types (multi-label, e.g. "DRUG|DEVICE") -> one 0/1 column per type
       - counts -> log(1 + x); missing numbers -> median + missing-indicator column
       - all numeric -> standardized
     Every transformation is FIT ON TRAIN ONLY, then applied to test (no leakage).
  4. Saves the matrices to data/processed/ for the model scripts.

"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# the 22 raw features
CATEGORICAL = [
    "phase", "allocation", "intervention_model", "primary_purpose", "masking",
    "gender", "healthy_volunteers", "lead_sponsor_class", "has_dmc",
    "is_fda_regulated_drug", "is_fda_regulated_device",
]
MULTI_LABEL = "intervention_types"          # 12th categorical feature
COUNT_FEATURES = [                          # skewed -> log(1 + x)
    "number_of_arms", "n_sites", "n_countries", "n_collaborators",
    "n_interventions", "n_conditions", "criteria_length",
]
OTHER_NUMERIC = ["min_age_years", "max_age_years", "start_year"]
ENROLLMENT = "enrollment"                    # ablation only (leaks the label)

MAX_AGE_NO_LIMIT = 100.0   # value used when a trial has no upper age limit


def prepare_frame(df):
    """Pandas-level cleaning that does not learn anything from the data."""
    df = df.copy()
    # Categorical: store as strings, missing becomes its own category.
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.upper().fillna("MISSING")
    # Ages: a missing maximum age means "no upper limit", not unknown.
    df["max_age_no_limit"] = df["max_age_years"].isna().astype(int)
    df["max_age_years"] = df["max_age_years"].fillna(MAX_AGE_NO_LIMIT)
    df[MULTI_LABEL] = df[MULTI_LABEL].fillna("NONE")
    return df


def multi_hot(series, vocabulary):
    """Expand 'DRUG|DEVICE' strings into 0/1 columns using a fixed vocabulary."""
    dummies = series.str.get_dummies(sep="|")
    dummies = dummies.reindex(columns=vocabulary, fill_value=0)
    dummies.columns = [f"intervention_{c.lower()}" for c in vocabulary]
    return dummies


def build_transformer(count_cols, numeric_cols):
    log_pipeline = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("count", log_pipeline, count_cols),
        ("num", num_pipeline, numeric_cols),
        ("flags", "passthrough", ["max_age_no_limit"]),
    ], verbose_feature_names_out=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cohort.csv")
    ap.add_argument("--with-enrollment", action="store_true",
                    help="add enrollment as a feature (leakage ablation)")
    args = ap.parse_args()

    out_dir = "data/processed_with_enrollment" if args.with_enrollment else "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    df = prepare_frame(pd.read_csv(args.input))
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    count_cols = COUNT_FEATURES + ([ENROLLMENT] if args.with_enrollment else [])
    transformer = build_transformer(count_cols, OTHER_NUMERIC)

    # Fit on train only, then apply to test.
    X_train_main = transformer.fit_transform(train)
    X_test_main = transformer.transform(test)

    vocab = sorted(set("|".join(train[MULTI_LABEL]).split("|")) - {"NONE"})
    X_train_int = multi_hot(train[MULTI_LABEL], vocab)
    X_test_int = multi_hot(test[MULTI_LABEL], vocab)

    X_train = np.hstack([X_train_main, X_train_int.values]).astype(np.float32)
    X_test = np.hstack([X_test_main, X_test_int.values]).astype(np.float32)
    y_train = train["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)
    feature_names = list(transformer.get_feature_names_out()) + list(X_train_int.columns)

    np.savez_compressed(os.path.join(out_dir, "data.npz"),
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        ids_train=train["nct_id"].to_numpy(), ids_test=test["nct_id"].to_numpy())
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=1)
    joblib.dump({"transformer": transformer, "intervention_vocab": vocab},
                os.path.join(out_dir, "preprocessor.joblib"))

    # summary
    print("PREPROCESSING SUMMARY")
    print(f"Setting           : {'WITH enrollment (ablation)' if args.with_enrollment else 'main (no enrollment)'}")
    print(f"Raw features      : {len(CATEGORICAL) + 1} categorical, "
          f"{len(count_cols) + len(OTHER_NUMERIC)} numeric")
    print(f"Feature dimension : {X_train.shape[1]} (after one-hot encoding and missing indicators)")
    print(f"Train             : {X_train.shape[0]:,} trials, {y_train.mean():.1%} terminated")
    print(f"Test              : {X_test.shape[0]:,} trials, {y_test.mean():.1%} terminated")
    print(f"NaNs left         : {int(np.isnan(X_train).sum() + np.isnan(X_test).sum())}")
    print(f"Saved to          : {out_dir}/")


if __name__ == "__main__":
    main()