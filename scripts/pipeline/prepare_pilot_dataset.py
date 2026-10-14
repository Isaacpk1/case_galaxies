"""Build the time-aware, tabular dataset used by the pilot baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "moda": "20260928T213206Z",
    "automotivo": "20260928T212753Z",
}
PILOT_CHANNELS = {
    "C&A Brasil",
    "EstiloRenner",
    "Hering Oficial",
    "Steal The Look",
    "Mandy Candy",
    "NiinaSecrets",
    "Edi Ribeiro",
    "Giuliana Mafra",
    "Ingryd Lourenco",
    "Maria Bardela",
    "Matando Matheus a Grito",
    "Vique",
    "BYD Auto Brasil",
    "Chevrolet Brasil",
    "Fiat Automoveis Brasil",
    "Jeep Brasil",
    "Nissan Brasil",
    "Renault Brasil",
    "Toyota do Brasil",
    "ComprecarTV",
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
    "title_word_count",
    "description_word_count",
    "has_question",
    "exclamation_count",
    "emoji_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "digit_count",
    "uppercase_ratio",
    "has_cta",
    "title_tag_overlap",
    "duration_bucket",
    "publish_hour_sin",
    "publish_hour_cos",
    "publish_weekday_sin",
    "publish_weekday_cos",
    "is_weekend",
    "channel_prior_mean_likes",
    "channel_prior_like_std",
    "channel_recent_5_median_likes",
    "channel_recent_10_median_likes",
    "channel_recent_5_like_trend",
    "channel_recent_5_like_volatility",
    "channel_prior_median_views",
    "channel_prior_median_comments",
    "channel_prior_like_view_rate",
]


def tag_count(value: object) -> int:
    try:
        return len(json.loads(value)) if isinstance(value, str) else 0
    except json.JSONDecodeError:
        return 0


def text_tokens(value: object) -> list[str]:
    return re.findall(r"[\wÀ-ÿ]+", str(value or "").lower())


def emoji_count(value: object) -> int:
    return len(re.findall(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", str(value or "")))


def uppercase_ratio(value: object) -> float:
    letters = [char for char in str(value or "") if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def has_cta(value: object) -> int:
    pattern = r"\b(curta|like|comente|comentem|compartilhe|salve|veja ate o fim)\b"
    return int(bool(re.search(pattern, str(value or "").lower())))


def add_time_aware_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True).copy()
    data["channel_prior_posts"] = data.groupby("channel_id").cumcount()
    data["channel_prior_median_likes"] = np.nan
    data["days_since_previous_post"] = np.nan

    for _, index in data.groupby("channel_id", sort=False).groups.items():
        channel = data.loc[index]
        # Videos published at the exact same instant cannot be used as each
        # other's history.  Compute the value at the first row of a timestamp
        # group, then carry it to the simultaneous rows.
        first_at_timestamp = ~channel["published_at"].duplicated()

        def strictly_prior(values: pd.Series) -> pd.Series:
            return values.where(first_at_timestamp).ffill()

        data.loc[index, "channel_prior_posts"] = strictly_prior(
            pd.Series(np.arange(len(channel)), index=channel.index, dtype=float)
        ).to_numpy()
        prior_likes = channel["like_count"].shift()
        prior_log_likes = np.log1p(prior_likes)
        data.loc[index, "channel_prior_median_likes"] = strictly_prior(prior_likes.expanding().median()).to_numpy()
        data.loc[index, "channel_prior_mean_likes"] = strictly_prior(prior_likes.expanding().mean()).to_numpy()
        data.loc[index, "channel_prior_like_std"] = strictly_prior(prior_likes.expanding().std()).to_numpy()
        data.loc[index, "channel_recent_5_median_likes"] = strictly_prior(prior_likes.rolling(5, min_periods=1).median()).to_numpy()
        data.loc[index, "channel_recent_10_median_likes"] = strictly_prior(prior_likes.rolling(10, min_periods=1).median()).to_numpy()
        data.loc[index, "channel_recent_5_like_trend"] = strictly_prior(prior_log_likes - prior_log_likes.rolling(5, min_periods=1).mean()).to_numpy()
        data.loc[index, "channel_recent_5_like_volatility"] = strictly_prior(prior_log_likes.rolling(5, min_periods=2).std()).to_numpy()
        prior_views = channel["view_count"].shift()
        prior_comments = channel["comment_count"].shift()
        data.loc[index, "channel_prior_median_views"] = strictly_prior(prior_views.expanding().median()).to_numpy()
        data.loc[index, "channel_prior_median_comments"] = strictly_prior(prior_comments.expanding().median()).to_numpy()
        prior_rate = (prior_likes + 1) / (prior_views + 1)
        data.loc[index, "channel_prior_like_view_rate"] = strictly_prior(prior_rate.expanding().median()).to_numpy()
        data.loc[index, "days_since_previous_post"] = strictly_prior(
            channel["published_at"].diff().dt.total_seconds().div(86_400)
        ).to_numpy()

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
    combined_text = data["title"].fillna("") + " " + data["description"].fillna("")
    data["title_word_count"] = data["title"].map(lambda value: len(text_tokens(value)))
    data["description_word_count"] = data["description"].map(lambda value: len(text_tokens(value)))
    data["has_question"] = combined_text.str.contains(r"\?", regex=True).astype(int)
    data["exclamation_count"] = combined_text.str.count("!")
    data["emoji_count"] = combined_text.map(emoji_count)
    data["hashtag_count"] = combined_text.str.count(r"#\w+")
    data["mention_count"] = combined_text.str.count(r"@\w+")
    data["url_count"] = combined_text.str.count(r"https?://\S+|www\.\S+")
    data["digit_count"] = combined_text.str.count(r"\d")
    data["uppercase_ratio"] = combined_text.map(uppercase_ratio)
    data["has_cta"] = combined_text.map(has_cta)
    data["title_tag_overlap"] = data.apply(
        lambda row: len(set(text_tokens(row["title"])) & set(json.loads(row["tags"])))
        / max(1, len(set(text_tokens(row["title"])))),
        axis=1,
    )
    data["duration_bucket"] = pd.cut(
        data["duration_seconds"],
        bins=[0, 15, 30, 60, 180],
        labels=["01_15", "16_30", "31_60", "61_180"],
        include_lowest=True,
    ).astype("string")
    data["publish_hour_sin"] = np.sin(2 * np.pi * data["publish_hour_utc"] / 24)
    data["publish_hour_cos"] = np.cos(2 * np.pi * data["publish_hour_utc"] / 24)
    data["publish_weekday_sin"] = np.sin(2 * np.pi * data["publish_weekday"] / 7)
    data["publish_weekday_cos"] = np.cos(2 * np.pi * data["publish_weekday"] / 7)
    data["is_weekend"] = data["publish_weekday"].ge(5).astype(int)
    data["channel_prior_like_std"] = data["channel_prior_like_std"].fillna(0)
    data["channel_recent_5_like_volatility"] = data["channel_recent_5_like_volatility"].fillna(0)
    data["is_model_ready"] = data["channel_prior_posts"].ge(MIN_PRIOR_POSTS)
    return data


def add_temporal_split(data: pd.DataFrame) -> pd.DataFrame:
    """Reserve the most recent 20% of every channel for final evaluation."""
    data = data.sort_values(["channel_id", "published_at", "video_id"]).copy()
    data["split"] = "train"
    for index in data.groupby("channel_id", sort=False).groups.values():
        channel = data.loc[index].sort_values(["published_at", "video_id"])
        test_size = int(np.ceil(len(channel) * TEST_FRACTION))
        test_start = channel.iloc[-test_size]["published_at"]
        data.loc[channel.index[channel["published_at"].ge(test_start)], "split"] = "test"
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
    data["view_count"] = pd.to_numeric(data["view_count"], errors="coerce")
    data["comment_count"] = pd.to_numeric(data["comment_count"], errors="coerce")
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
