from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import pickle

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
DATA_DIR = MODEL_DIR / "dataset"
OUTPUT_DIR = MODEL_DIR / "sample_data"


def load_pickle(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with path.open("rb") as f:
        return pickle.load(f)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def landinggear_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "LandingGear_Balanced_Dataset.csv")
    df["Stiffness_Damping_Product"] = df["K_Stiffness"] * df["B_Damping"]
    return df


def aerospace_feature_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "aerospace_structural_design_dataset.csv")
    cols = df.columns.tolist()

    df["Durability_bin"] = df["Durability"].map({"Low": 0, "Medium": 1, "High": 1})
    df["Life_to_Load"] = df[cols[9]] / df[cols[4]]
    df["Strength_x_Thick"] = df[cols[4]] * df[cols[12]]
    df["Stiffness_Density"] = df[cols[1]] / df[cols[3]]
    df["Temp_Thickness"] = df[cols[7]] * df[cols[12]]

    cat_features = ["Material Type", "Structural Shape", "Load Distribution", "Vibration Damping"]
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

    numeric_features = [
        cols[1],
        cols[3],
        cols[4],
        cols[7],
        cols[12],
        "Life_to_Load",
        "Strength_x_Thick",
        "Stiffness_Density",
        "Temp_Thickness",
    ]
    dummy_features = [
        c
        for c in df_encoded.columns
        if c.startswith(
            (
                "Material Type_",
                "Structural Shape_",
                "Load Distribution_",
                "Vibration Damping_",
            )
        )
    ]
    feature_cols = numeric_features + dummy_features
    return df_encoded[feature_cols]


def resolve_feature_names(model: Any, fallback: List[str] | None = None) -> List[str] | None:
    if isinstance(model, dict) and "feature_names" in model:
        return list(model["feature_names"])
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if fallback is not None:
        return list(fallback)
    return None


def sample_rows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42)


def align_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[list(columns)]


def build_lstm_sample(seq_length: int = 50) -> Tuple[pd.DataFrame, np.ndarray]:
    index_cols = ["UnitNumber", "Cycle"]
    op_cols = [f"OpSet{i}" for i in range(1, 4)]
    sensor_cols = [f"Sensor{i}" for i in range(1, 22)]
    cols = index_cols + op_cols + sensor_cols

    train_path = DATA_DIR / "CMAPSSData" / "train_FD001.txt"
    train = pd.read_csv(train_path, sep=" ", header=None)
    train = train.dropna(axis=1)
    train.columns = cols

    max_cycle = train.groupby("UnitNumber")["Cycle"].max().reset_index()
    max_cycle.columns = ["UnitNumber", "MaxCycle"]
    train = train.merge(max_cycle, on="UnitNumber")
    train["RUL"] = train["MaxCycle"] - train["Cycle"]
    train = train.drop(columns=["MaxCycle"])

    cols_to_drop = [
        "OpSet3",
        "Sensor1",
        "Sensor5",
        "Sensor6",
        "Sensor10",
        "Sensor14",
        "Sensor16",
        "Sensor18",
        "Sensor19",
    ]
    train = train.drop(columns=cols_to_drop)

    feats = train.columns.drop(["UnitNumber", "Cycle", "RUL"])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train[feats] = scaler.fit_transform(train[feats])

    unit_id = int(train["UnitNumber"].iloc[0])
    unit_df = train[train["UnitNumber"] == unit_id].copy()
    unit_df = unit_df.sort_values("Cycle")

    seq = unit_df[feats].values
    if seq.shape[0] >= seq_length:
        seq = seq[-seq_length:]
    else:
        pad = np.zeros((seq_length - seq.shape[0], seq.shape[1]))
        seq = np.vstack([pad, seq])

    seq_df = pd.DataFrame(seq, columns=feats)
    seq_array = np.expand_dims(seq, axis=0)
    return seq_df, seq_array


def main() -> None:
    ensure_output_dir()

    landing_df = landinggear_dataframe()
    aerospace_df = aerospace_feature_dataframe()

    pkl_files = sorted(MODEL_DIR.glob("*.pkl"))
    outputs: Dict[str, Path] = {}

    for path in pkl_files:
        model = load_pickle(path)
        name = path.stem

        fallback_features: List[str] | None = None
        if "durability" in name.lower():
            fallback_features = list(aerospace_df.columns)
        elif "landinggear" in name.lower() or "fault" in name.lower():
            fallback_features = [
                "RunID",
                "Max_Deflection",
                "Max_Velocity",
                "Settling_Time",
                "Mass",
                "K_Stiffness",
                "B_Damping",
                "Fault_Code",
                "RUL",
                "Stiffness_Damping_Product",
            ]
        elif "engine" in name.lower():
            fallback_features = None

        feature_names = resolve_feature_names(model, fallback_features)
        if feature_names is None:
            if hasattr(model, "n_features_in_"):
                feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
            else:
                feature_names = []

        if "durability" in name.lower():
            sample_df = sample_rows(aerospace_df)
        else:
            sample_df = sample_rows(landing_df)

        if "Stiffness_Damping_Product" in feature_names and "Stiffness_Damping_Product" not in sample_df.columns:
            if {"K_Stiffness", "B_Damping"}.issubset(sample_df.columns):
                sample_df["Stiffness_Damping_Product"] = (
                    sample_df["K_Stiffness"] * sample_df["B_Damping"]
                )

        sample_df = align_columns(sample_df, feature_names)
        out_path = OUTPUT_DIR / f"{name}_sample.csv"
        sample_df.to_csv(out_path, index=False)
        outputs[name] = out_path

    lstm_df, lstm_array = build_lstm_sample(seq_length=50)
    lstm_csv = OUTPUT_DIR / "remainingUsefulLife_lstm_sequence.csv"
    lstm_npy = OUTPUT_DIR / "remainingUsefulLife_lstm_sequence.npy"
    lstm_df.to_csv(lstm_csv, index=False)
    np.save(lstm_npy, lstm_array)
    outputs["remainingUsefulLife_lstm"] = lstm_csv

    print("Sample data generated:")
    for model_name, out_path in outputs.items():
        print(f"- {model_name}: {out_path}")
    print(f"- remainingUsefulLife_lstm (numpy): {lstm_npy}")


if __name__ == "__main__":
    main()
