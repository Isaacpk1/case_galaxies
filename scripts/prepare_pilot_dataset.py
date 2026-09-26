"""Build the time-aware, tabular dataset used by the pilot baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "moda": "20260924T163050Z",
    "automotivo": "20260924T163415Z",
}
PILOT_CHANNELS = {
    "C&A Brasil",
    "EstiloRenner",
    "Hering Oficial",
    "Steal The Look",
    "Mandy Candy",
    "NiinaSecrets",
    "Hyundai Motor Brasil",
    "Honda Automoveis",
    "Volkswagen do Brasil",
    "Acelerados",
    "Carros com Camanzi",
    "De Carona com Leandro",
}
MIN_PRIOR_POSTS = 5
TEST_FRACTION = 0.20

# This is the contract consumed by the first tabular experiment.  Keep targets
# separate from features: they are present for supervised training/evaluation,
# but never available for an unpublished creative at prediction time.
CONSUMPTION_COLUMNS = [
    "video_id",
    "channel_id",
    "channel_name",
    "sector",
    "market",
    "channel_type",
    "published_at",
    "duration_seconds",
    "channel_prior_posts",
    "channel_prior_median_likes",
    "days_since_previous_post",
    "publish_hour_utc",
    "publish_weekday",
    "title_length",
    "description_length",
    "tag_count",
    "baseline_log_likes",
    "target_log_likes",
    "target_relative_to_prior_median",
    "split",
]


def tag_count(value: object) -> int:
    try:
        return len(json.loads(value)) if isinstance(value, str) else 0
    except json.JSONDecodeError:
        return 0


def add_time_aware_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True).copy()
    data["channel_prior_posts"] = data.groupby("channel_id").cumcount()
    data["channel_prior_median_likes"] = np.nan
    data["days_since_previous_post"] = np.nan

    for _, index in data.groupby("channel_id", sort=False).groups.items():
        channel = data.loc[index]
        data.loc[index, "channel_prior_median_likes"] = channel["like_count"].shift().expanding().median().to_numpy()
        data.loc[index, "days_since_previous_post"] = channel["published_at"].diff().dt.total_seconds().div(86_400).to_numpy()

    data["target_log_likes"] = np.log1p(data["like_count"])
    data["target_relative_to_prior_median"] = (
        (data["like_count"] + 1) / (data["channel_prior_median_likes"] + 1)
    )
    data["baseline_log_likes"] = np.log1p(data["channel_prior_median_likes"])
    data["publish_hour_utc"] = data["published_at"].dt.hour
    data["publish_weekday"] = data["published_at"].dt.dayofweek
    data["title_length"] = data["title"].fillna("").str.len()
    data["description_length"] = data["description"].fillna("").str.len()
    data["tag_count"] = data["tags"].map(tag_count)
    data["is_model_ready"] = data["channel_prior_posts"].ge(MIN_PRIOR_POSTS)
    return data


def add_temporal_split(data: pd.DataFrame) -> pd.DataFrame:
    """Reserve the most recent 20% of every channel for final evaluation."""
    data = data.sort_values(["channel_id", "published_at", "video_id"]).copy()
    data["split"] = "train"
    for index in data.groupby("channel_id", sort=False).groups.values():
        test_size = int(np.ceil(len(index) * TEST_FRACTION))
        data.loc[index[-test_size:], "split"] = "test"
    return data


def main() -> None:
    parts = []
    for run_id in RUNS.values():
        path = ROOT / "data" / "silver" / "data_discovery" / run_id / "eligible_shorts.csv"
        parts.append(pd.read_csv(path))

    raw = pd.concat(parts, ignore_index=True)
    data = raw.loc[raw["channel_name"].isin(PILOT_CHANNELS)].copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True, errors="raise")
    data["like_count"] = pd.to_numeric(data["like_count"], errors="raise")
    data = data.drop_duplicates(subset="video_id", keep="first")
    data = add_time_aware_features(data)

    gold_dir = ROOT / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    full_path = gold_dir / "pilot_dataset.csv"
    ready_path = gold_dir / "pilot_model_ready.csv"
    consumption_path = gold_dir / "pilot_consumption_table.csv"
    data.to_csv(full_path, index=False)
    data.loc[data["is_model_ready"]].to_csv(ready_path, index=False)
    model_ready = data.loc[data["is_model_ready"]].copy()
    model_ready = add_temporal_split(model_ready)
    consumption = model_ready.loc[:, CONSUMPTION_COLUMNS].copy()
    consumption.to_csv(consumption_path, index=False)

    print(f"Pilot rows: {len(data)}")
    print(f"Model-ready rows: {int(data['is_model_ready'].sum())}")
    print(f"Train rows: {int(consumption['split'].eq('train').sum())}")
    print(f"Test rows: {int(consumption['split'].eq('test').sum())}")
    print(f"Pilot channels: {data['channel_id'].nunique()}")
    print(f"Saved: {full_path}")
    print(f"Saved: {ready_path}")
    print(f"Saved: {consumption_path}")


if __name__ == "__main__":
    main()
