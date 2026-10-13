"""Testes da tool: contrato, antivazamento, consistencia do modelo e ranking.

    PYTHONPATH=src python -m pytest -q tests/test_tool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from galaxies_agent import ChannelStats, CreativeInput, Predictor
from galaxies_agent.predictor import ROOT


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor()


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / latest["path"], parse_dates=["published_at"])
    return frame[frame["split"] == "train"]


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / "20261002T013000Z" / "eligible_shorts.csv")


def as_input(row, silver_row, creative: str | None = None) -> CreativeInput:
    tags = json.loads(silver_row["tags"]) if isinstance(silver_row["tags"], str) else []
    return CreativeInput(creative=creative or row.video_id, caption=str(silver_row["title"] or ""),
                         description="" if pd.isna(silver_row["description"]) else str(silver_row["description"]),
                         tags=tags, planned_datetime=row.published_at.to_pydatetime(), channel_id=row.channel_id,
                         sector=row.sector, duration_seconds=float(row.duration_seconds))


def test_contract_requires_channel():
    with pytest.raises(ValidationError):
        CreativeInput(caption="x", planned_datetime="2026-10-20T18:00", sector="moda")


def test_no_leakage_features_match_gold(predictor, gold, silver):
    """Recalcular as features na data de publicacao deve reproduzir o Gold: so posts estritamente anteriores."""
    sil = silver.set_index("video_id")
    sample = gold.sample(60, random_state=1)
    for row in sample.itertuples():
        features, _ = predictor.build_row(as_input(row, sil.loc[row.video_id]))
        assert features["baseline_log_likes"] == pytest.approx(row.baseline_log_likes, abs=1e-9)
        assert features["channel_prior_posts"] == row.channel_prior_posts
        assert features["days_since_previous_post"] == pytest.approx(row.days_since_previous_post, abs=1e-6)
        assert features["publish_hour_utc"] == str(row.publish_hour_utc)
        assert features["title_length"] == row.title_length
        assert features["tag_count"] == row.tag_count


def test_prediction_contract_and_shap(predictor, gold, silver):
    sil = silver.set_index("video_id")
    row = next(gold.sample(1, random_state=3).itertuples())
    item = as_input(row, sil.loc[row.video_id])
    out = predictor.predict(item)
    assert out.embedding_source == "cache"
    assert out.interval_low <= out.predicted_metric <= out.interval_high
    assert out.top_drivers and len(out.comparables) == 3
    assert row.video_id not in {c.video_id for c in out.comparables}  # o proprio Short nao e comparavel de si mesmo
    # SHAP: soma das contribuicoes + valor esperado = desvio previsto
    features, _ = predictor.build_row(item)
    vector, _ = predictor.embedder.embed(item.creative)
    pcs = (vector - predictor.pca_mean) @ predictor.pca_components.T
    pool = predictor._pool(features, True, pcs)
    shap = predictor.model_emb.get_feature_importance(pool, type="ShapValues")[0]
    assert shap.sum() == pytest.approx(predictor.model_emb.predict(pool)[0], abs=1e-4)


def test_fallback_without_image_is_low_confidence(predictor, gold, silver):
    sil = silver.set_index("video_id")
    row = next(gold.sample(1, random_state=4).itertuples())
    item = as_input(row, sil.loc[row.video_id]).model_copy(update={"creative": None})
    out = predictor.predict(item)
    assert out.confidence_flag == "baixa"
    assert out.embedding_source == "sem_imagem"
    assert out.comparables == []


def test_external_channel_stats(predictor):
    item = CreativeInput(caption="Novo look de verao", planned_datetime="2026-10-20T18:00", sector="moda",
                         channel_stats=ChannelStats(channel_name="Marca X", channel_type="brand",
                                                    prior_likes=[120, 300, 90, 210, 150] * 8,
                                                    last_post_at="2026-10-15T12:00"))
    out = predictor.predict(item)
    assert out.channel_baseline_likes == 150
    assert any("fora da base" in r for r in out.confidence_reasons)


def test_ranking_is_sorted(predictor, gold, silver):
    sil = silver.set_index("video_id")
    channel = gold["channel_id"].value_counts().index[0]
    rows = gold[gold["channel_id"] == channel].tail(3)
    when = rows["published_at"].max().to_pydatetime()
    items = [as_input(r, sil.loc[r.video_id]).model_copy(update={"planned_datetime": when}) for r in rows.itertuples()]
    out = predictor.rank(items)
    rel = [r.relative_to_channel for r in out.ranking]
    assert rel == sorted(rel, reverse=True)
    assert [r.rank for r in out.ranking] == [1, 2, 3]


def test_no_outcome_columns_in_model(predictor):
    names = set(predictor.model_emb.feature_names_) | set(predictor.model_base.feature_names_)
    assert not names & {"like_count", "view_count", "comment_count"}
    assert not any(n.startswith("target_") for n in names)
