"""Data loading, cleaning, and feature engineering — dataset-agnostic."""

import os
import numpy as np
import pandas as pd
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, CORRELATION_THRESHOLD, RANDOM_STATE, TEST_SIZE, get_dataset_config,
)


def load_data(data_dir: str = None, dataset_name: str = None) -> tuple[pd.DataFrame, str]:
    """Load and concatenate CSV files for the specified dataset."""
    ds = get_dataset_config(dataset_name)
    label_col = ds["label_column"]

    if data_dir is None:
        data_dir = DATA_DIR

    print(f"Dataset: {ds['description']}")

    frames = []
    # Try to find CSVs — search data_dir and one level of subdirectories
    search_dirs = [data_dir]
    for item in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        sub = os.path.join(data_dir, item)
        if os.path.isdir(sub):
            search_dirs.append(sub)

    for csv_file in ds["csv_files"]:
        loaded = False
        for search_dir in search_dirs:
            path = os.path.join(search_dir, csv_file)
            if os.path.exists(path):
                df = pd.read_csv(path, low_memory=False)
                df_sampled = df.sample(frac=ds["sample_fraction"], random_state=RANDOM_STATE)
                frames.append(df_sampled)
                print(f"  Loaded {csv_file}: {len(df_sampled):,} rows (sampled from {len(df):,})")
                loaded = True
                break
        if not loaded:
            print(f"  WARN: {csv_file} not found, skipping")

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found. Place dataset files in {data_dir}\n"
            f"Expected files: {ds['csv_files'][:3]}..."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Drop columns specified in config (e.g., 'id', 'attack_cat')
    drop_cols = ds.get("drop_columns", [])
    existing_drop = [c for c in drop_cols if c in combined.columns]
    if existing_drop:
        combined = combined.drop(columns=existing_drop)
        print(f"Dropped columns: {existing_drop}")

    print(f"\nTotal rows after sampling: {len(combined):,}")
    return combined, label_col


def clean_data(df: pd.DataFrame, label_col: str, min_samples: int) -> pd.DataFrame:
    """Remove duplicates, NaN, inf values and filter small classes."""
    initial = len(df)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    label_col = label_col.strip()

    df = df.drop_duplicates(keep="first")
    df = df.dropna()

    # Replace inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Filter small classes
    if min_samples > 0:
        df = df.groupby(label_col).filter(lambda x: len(x) > min_samples)

    # Encode non-numeric columns (except label)
    for col in df.select_dtypes(include=["object"]).columns:
        if col != label_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Downcast for memory efficiency
    int_cols = df.select_dtypes(include=["int64"]).columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    print(f"Cleaned: {initial:,} -> {len(df):,} rows")
    print(f"Classes: {df[label_col].value_counts().to_dict()}")
    return df, label_col


def remove_correlated_features(df: pd.DataFrame, label_col: str,
                                threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """Drop features with correlation above threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    # Never drop the label
    to_drop = [c for c in to_drop if c != label_col]

    df = df.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} correlated features (threshold={threshold})")
    return df


def prepare_splits(df: pd.DataFrame, label_col: str):
    """Split, undersample, scale, and return train/test sets + metadata."""
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Undersample to balance classes
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(f"After undersampling: {len(X_resampled):,} samples")

    # Z-score normalization
    for col in X_resampled.columns:
        X_resampled[col] = stats.zscore(X_resampled[col])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute NaN (from zscore of constant columns)
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_names = list(X_resampled.columns)
    class_names = sorted(y_resampled.unique().tolist())

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {class_names}")

    return X_train, X_test, y_train, y_test, feature_names, class_names, scaler, imputer


def run_preprocessing(data_dir: str = None, dataset_name: str = None):
    """Full preprocessing pipeline. Returns all artifacts needed for training."""
    ds = get_dataset_config(dataset_name)

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df, label_col = load_data(data_dir, dataset_name)

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning data")
    print("=" * 60)
    df, label_col = clean_data(df, label_col, ds["min_samples_per_class"])

    print("\n" + "=" * 60)
    print("STEP 3: Removing correlated features")
    print("=" * 60)
    df = remove_correlated_features(df, label_col)

    print("\n" + "=" * 60)
    print("STEP 4: Preparing train/test splits")
    print("=" * 60)
    return prepare_splits(df, label_col)
