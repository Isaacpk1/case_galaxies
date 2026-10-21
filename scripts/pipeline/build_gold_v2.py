"""Gold v2: automotivo + moda, canais equilibrados (pessoa / marca / midia), ate 150 Shorts por canal.

Mesmo contrato do Gold v1 (prepare_pilot_dataset.add_time_aware_features + alvo residual + split temporal
por canal), com duas diferencas:
  - o historico do canal (B0, posts anteriores, dias desde o anterior) usa TODOS os Shorts elegiveis
    coletados do canal, nao so os que entram no treino (o teto de 150 vale para as linhas de treino);
  - colunas novas: `grupo` (pessoa | marca | midia) e `origem_v2`:
        v1            ja estava no Gold v1 (o modelo servido treinou nesta linha)
        novo_periodo  canal do v1, publicado depois do ultimo Short do canal no Gold v1 (nunca visto)
        backfill      canal do v1, Short antigo que o v1 nao tinha (teto de 300 uploads)
        novo_canal    canal que nao existia no v1
    `avaliacao_congelada` = novo_periodo ou novo_canal: linhas onde o modelo servido (treinado no Gold v1)
    pode ser avaliado sem vazamento.

Entradas: data/silver/eligible_shorts/<v2auto_*>/history_all_eligible.csv e <v2moda_*>/history_all_eligible.csv
Saida:    data/gold/snapshots/gold_v2_<hash>/training_snapshot.csv + manifest.json (LATEST.json nao muda)

    python scripts/pipeline/build_gold_v2.py --silver-ids v2auto_20261020T232948Z v2moda_20261021T000925Z
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pilot_dataset import add_time_aware_features  # noqa: E402

GOLD_V1 = ROOT / "data" / "gold" / "snapshots" / "gold_v1_20261002T013000Z_663f502d" / "training_snapshot.csv"
MIN_PRIOR_POSTS = 5
TEST_FRACTION = 0.20
FEATURES = ["duration_seconds", "channel_prior_posts", "channel_prior_median_likes", "baseline_log_likes",
            "days_since_previous_post", "title_length", "description_length", "tag_count",
            "publish_hour_utc", "publish_weekday", "sector", "channel_type", "market"]
FORBIDDEN = {"like_count", "view_count", "comment_count"}


def temporal_split(data: pd.DataFrame, test_fraction: float) -> pd.Series:
    split = pd.Series("train", index=data.index)
    for _, channel in data.sort_values(["published_at", "video_id"]).groupby("channel_id"):
        test_size = int(np.ceil(len(channel) * test_fraction))
        test_start = channel.iloc[-test_size]["published_at"]
        split.loc[channel.index[channel["published_at"].ge(test_start)]] = "test"
    return split


def check_baseline(gold: pd.DataFrame, history: pd.DataFrame, sample: int = 500) -> dict:
    rng = np.random.default_rng(7)
    rows = gold.iloc[rng.choice(len(gold), size=min(sample, len(gold)), replace=False)]
    by_channel = {k: g for k, g in history.groupby("channel_id")}
    worst = 0.0
    for row in rows.itertuples(index=False):
        h = by_channel[row.channel_id]
        prior = h.loc[h["published_at"] < row.published_at, "like_count"]
        worst = max(worst, abs(float(np.log1p(prior.median())) - row.baseline_log_likes))
    return {"rows_checked": int(len(rows)), "max_abs_error": round(worst, 9), "passed": bool(worst < 1e-6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--silver-ids", nargs="+", required=True)
    args = parser.parse_args()

    parts = []
    for sid in args.silver_ids:
        frame = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / sid / "history_all_eligible.csv",
                            dtype={"video_id": "string"})
        frame["silver_id"] = sid
        parts.append(frame)
    history = pd.concat(parts, ignore_index=True).drop_duplicates("video_id")
    history["published_at"] = pd.to_datetime(history["published_at"], utc=True)
    for column in ("like_count", "view_count", "comment_count"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["like_count"])
    history["in_training_cap"] = history["in_training_cap"].astype(str).str.lower().eq("true")

    data = add_time_aware_features(history)
    gold = data.loc[data["in_training_cap"] & data["channel_prior_posts"].ge(MIN_PRIOR_POSTS)].copy()
    gold["target_log_likes"] = np.log1p(gold["like_count"])
    gold["target_residual_log_likes"] = gold["target_log_likes"] - gold["baseline_log_likes"]
    gold["target_relative_to_prior_median"] = (gold["like_count"] + 1) / (gold["channel_prior_median_likes"] + 1)
    gold["split"] = temporal_split(gold, TEST_FRACTION)

    v1 = pd.read_csv(GOLD_V1, usecols=["video_id", "channel_id", "published_at"], dtype={"video_id": "string"})
    v1["published_at"] = pd.to_datetime(v1["published_at"], utc=True)
    v1_last = v1.groupby("channel_id")["published_at"].max()
    in_v1 = gold["video_id"].isin(set(v1["video_id"]))
    v1_channel = gold["channel_id"].isin(v1_last.index)
    after = gold["published_at"] > gold["channel_id"].map(v1_last)
    gold["origem_v2"] = np.select([in_v1, v1_channel & after, v1_channel], ["v1", "novo_periodo", "backfill"], "novo_canal")
    gold["avaliacao_congelada"] = gold["origem_v2"].isin(["novo_periodo", "novo_canal"])

    check = check_baseline(gold, history)
    order_ok = all(g.loc[g["split"].eq("train"), "published_at"].max() < g.loc[g["split"].eq("test"), "published_at"].min()
                   for _, g in gold.groupby("channel_id"))
    if not check["passed"] or not order_ok or FORBIDDEN & set(FEATURES):
        raise AssertionError(f"checagens de vazamento falharam: baseline={check} ordem={order_ok}")

    spine = ["video_id", "channel_id", "channel_name", "grupo", "published_at", "split", "origem_v2", "avaliacao_congelada"]
    targets = ["target_log_likes", "target_residual_log_likes", "target_relative_to_prior_median"]
    gold = gold[spine + targets + FEATURES + ["title", "silver_id"]].sort_values(["channel_id", "published_at", "video_id"])

    version_hash = hashlib.sha256(json.dumps(sorted(args.silver_ids)).encode()).hexdigest()[:8]
    version = f"gold_v2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{version_hash}"
    out = ROOT / "data" / "gold" / "snapshots" / version
    out.mkdir(parents=True, exist_ok=False)
    gold.to_csv(out / "training_snapshot.csv", index=False)

    def counts(frame: pd.DataFrame) -> dict:
        return {"linhas": int(len(frame)), "canais": int(frame["channel_id"].nunique())}

    manifest = {
        "dataset_version": version, "created_at": datetime.now(timezone.utc).isoformat(),
        "silver_ids": args.silver_ids, "cap_per_channel": 150, "min_prior_posts": MIN_PRIOR_POSTS,
        "test_fraction": TEST_FRACTION, "history_rows": int(len(history)), **counts(gold),
        "periodo": [str(gold["published_at"].min()), str(gold["published_at"].max())],
        "por_setor": {k: counts(g) for k, g in gold.groupby("sector")},
        "por_grupo": {k: counts(g) for k, g in gold.groupby("grupo")},
        "por_setor_grupo": {f"{s}/{g}": counts(f) for (s, g), f in gold.groupby(["sector", "grupo"])},
        "por_origem": {k: counts(g) for k, g in gold.groupby("origem_v2")},
        "split": gold["split"].value_counts().to_dict(),
        "avaliacao_congelada": counts(gold[gold["avaliacao_congelada"]]),
        "checagens": {"baseline_so_posts_anteriores": check, "treino_antes_do_teste_por_canal": order_ok,
                      "metricas_do_proprio_video_fora_das_features": True},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
