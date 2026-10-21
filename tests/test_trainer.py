"""Treino dinamico com base no esquema minimo do TAP.

    PYTHONPATH=src python -m pytest -q tests/test_trainer.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from galaxies_agent.predictor import ROOT
from galaxies_agent.trainer import build_features, load_client_base, train_client_model


@pytest.fixture(scope="module")
def client_base() -> pd.DataFrame:
    """Base 'de cliente' montada a partir dos canais de marca do Gold v1, renomeada para o esquema do TAP."""
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    gold = pd.read_csv(ROOT / latest["path"])
    silver = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / "20261002T013000Z" / "eligible_shorts.csv")
    brands = silver[silver["channel_type"] == "brand"]
    return pd.DataFrame({"post_id": brands["video_id"], "channel_id": brands["channel_id"], "published_at": brands["published_at"],
                         "caption": brands["title"], "metric_target": brands["like_count"], "sector": brands["sector"],
                         "channel_type": "brand", "duration_seconds": brands["duration_seconds"], "tags": brands["tags"],
                         "description": brands["description"], "coluna_extra": 1})


def test_rejects_missing_columns(client_base):
    with pytest.raises(ValueError, match="esquema minimo"):
        load_client_base(client_base.drop(columns=["metric_target"]))


def test_invalid_rows_are_dropped(client_base):
    bad = pd.concat([client_base.head(20), pd.DataFrame([{"post_id": "x", "channel_id": "c", "published_at": "nao e data",
                                                          "caption": "", "metric_target": -5, "sector": "moda"}])])
    data, dropped = load_client_base(bad)
    assert dropped == 1 and len(data) == 20


def test_baseline_uses_only_prior_posts(client_base):
    data, _ = load_client_base(client_base)
    feats = build_features(data)
    for row in feats.sample(40, random_state=3).itertuples():
        ch = data[data["channel_id"] == row.channel_id]
        prior = ch.loc[ch["published_at"] < row.published_at, "metric_target"]
        assert row.baseline_log_likes == pytest.approx(np.log1p(prior.median()))
        assert row.channel_prior_posts == len(prior)


def test_trains_and_reports(tmp_path, client_base):
    report = train_client_model(client_base, nome="teste", out_base=tmp_path)
    assert report.canais >= 10 and report.treino > report.teste > 0
    assert set(report.metricas_teste) == {"B0", "modelo"}
    assert 0 <= report.intervalo_80["cobertura_teste"] <= 1
    assert report.decisao and (tmp_path / report.modelo_dir.split("/")[-1] / "model.cbm").exists()
