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
    return Predictor(live_rubric=False)  # testes offline: rubrica so do cache da base


@pytest.fixture(scope="module")
def gold(predictor) -> pd.DataFrame:
    """Gold em que o pacote servido foi treinado (metadata do pacote), nao o LATEST dos experimentos."""
    frame = pd.read_csv(ROOT / predictor.meta["inputs"]["gold"]["path"], parse_dates=["published_at"], dtype={"video_id": str})
    return frame[frame["split"] == "train"]


@pytest.fixture(scope="module")
def silver(predictor) -> pd.DataFrame:
    """Titulo, descricao e tags dos Shorts: mesmas Silvers que montaram o historico do pacote servido."""
    inputs = predictor.meta["inputs"]
    base = ROOT / "data" / "silver" / "eligible_shorts"
    if "silver_history" not in inputs:  # pacote do Gold v1
        return pd.read_csv(base / "20261002T013000Z" / "eligible_shorts.csv", dtype={"video_id": str})
    parts = [pd.read_csv(base / sid / "history_all_eligible.csv", dtype={"video_id": str}) for sid in inputs["silver_history"]]
    extra = inputs.get("silver_history_v1_only_channels")
    if extra:
        parts.append(pd.read_csv(base / extra["silver_id"] / "eligible_shorts.csv", dtype={"video_id": str}))
    return pd.concat(parts, ignore_index=True).drop_duplicates("video_id")


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
    assert out.embedding_source == "cache" and out.rubric_source == "cache"
    assert out.model_variant == "final" and out.confidence_flag in ("alta", "baixa")
    assert out.video_attributes and "forca_hook_0a10" in out.video_attributes
    assert out.interval_low <= out.predicted_metric <= out.interval_high
    assert out.top_drivers and len(out.comparables) == 3
    assert row.video_id not in {c.video_id for c in out.comparables}  # o proprio Short nao e comparavel de si mesmo
    # SHAP: soma das contribuicoes + valor esperado = desvio previsto
    prepared = predictor.prepare(item)
    pool = predictor._pool(prepared["row"], "final")
    shap = predictor.models["final"].get_feature_importance(pool, type="ShapValues")[0]
    assert shap.sum() == pytest.approx(predictor.models["final"].predict(pool)[0], abs=1e-4)
    assert np.log(out.relative_to_channel) == pytest.approx(shap.sum(), abs=2e-3)


def test_features_match_training_for_base_short(predictor, gold, silver):
    """Comparaveis e rubrica recalculados na tool batem com o que o modelo viu no treino."""
    from galaxies_agent.features import KNN_COLS
    sil = silver.set_index("video_id")
    late = gold.sort_values("published_at").groupby("channel_id").tail(1).head(5)
    for row in late.itertuples():
        prepared = predictor.prepare(as_input(row, sil.loc[row.video_id]))
        assert prepared["variant"] == "final"
        assert not np.isnan(prepared["row"]["knn5_resid_mean"])  # canal com >= 5 Shorts anteriores
        assert prepared["row"]["gemini_coverage"] == 1.0
        assert set(KNN_COLS) <= set(prepared["row"])


def test_without_rubric_uses_embedding_fallback(predictor, gold, silver):
    sil = silver.set_index("video_id")
    row = next(gold.sample(1, random_state=5).itertuples())
    item = as_input(row, sil.loc[row.video_id])
    out = predictor.predict(item, attributes=(None, "sem_chave_gemini"))
    assert out.model_variant == "with_embedding"
    assert out.confidence_flag == "baixa"
    assert any("Gemini" in r for r in out.confidence_reasons)


def test_fallback_without_image_is_low_confidence(predictor, gold, silver):
    sil = silver.set_index("video_id")
    row = next(gold.sample(1, random_state=4).itertuples())
    item = as_input(row, sil.loc[row.video_id]).model_copy(update={"creative": None})
    out = predictor.predict(item)
    assert out.confidence_flag == "baixa"
    assert out.embedding_source == "sem_imagem"
    assert out.model_variant == "without_embedding"
    assert out.comparables == []


def test_external_channel_stats(predictor):
    item = CreativeInput(caption="Novo look de verao", planned_datetime="2026-10-20T18:00", sector="moda",
                         channel_stats=ChannelStats(channel_name="Marca X", channel_type="brand",
                                                    prior_likes=[120, 300, 90, 210, 150] * 8,
                                                    last_post_at="2026-10-15T12:00"))
    out = predictor.predict(item)
    assert out.channel_baseline_likes == 150
    assert any("fora da base" in r for r in out.confidence_reasons)


def test_subscribers_are_informative_only(predictor, gold, silver):
    """Inscritos entram no contrato (TAP) e voltam na saida, mas nao mudam a previsao."""
    stats = dict(channel_name="Marca X", channel_type="brand", prior_likes=[120, 300, 90, 210, 150] * 8,
                 last_post_at="2026-10-15T12:00")
    base = dict(caption="Novo look de verao", planned_datetime="2026-10-20T18:00", sector="moda")
    without = predictor.predict(CreativeInput(**base, channel_stats=ChannelStats(**stats)))
    with_subs = predictor.predict(CreativeInput(**base, channel_stats=ChannelStats(**stats, subscribers=250_000)))
    assert with_subs.channel_subscribers == 250_000 and without.channel_subscribers is None
    assert with_subs.predicted_metric == without.predicted_metric
    with pytest.raises(ValidationError):
        ChannelStats(**stats, subscribers=-1)
    if predictor.subscribers:  # canal da base: inscritos da ultima Silver channel_stats
        sil = silver.set_index("video_id")
        row = next(gold.sample(1, random_state=9).itertuples())
        out = predictor.predict(as_input(row, sil.loc[row.video_id]))
        assert out.channel_subscribers == predictor.subscribers.get(row.channel_id)


def test_channel_outside_training_keeps_history(predictor):
    """Canal com historico na tool mas fora do treino do pacote: preve pelo channel_id, com confianca baixa."""
    outside = predictor.channels[~predictor.channels["in_training"].astype(bool)]
    if outside.empty:
        pytest.skip("todos os canais do historico estao no treino deste pacote")
    channel_id = outside.index[0]
    last = predictor.history.loc[predictor.history["channel_id"] == channel_id, "published_at"].max()
    item = CreativeInput(caption="Teste", planned_datetime=(last + pd.Timedelta(days=1)).to_pydatetime(),
                         channel_id=channel_id, sector=outside.iloc[0]["sector"])
    out = predictor.predict(item)
    assert out.confidence_flag == "baixa"
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
    names = set().union(*(set(m.feature_names_) for m in predictor.models.values()))
    assert not names & {"like_count", "view_count", "comment_count"}
    assert not any(n.startswith("target_") for n in names)


def test_live_rubric_path_with_mocked_gemini(tmp_path, monkeypatch, predictor, gold, silver):
    """Criativo fora do cache: a tool chama o Gemini (simulado), valida o JSON, grava cache e usa o modelo final."""
    import io
    import urllib.request

    from galaxies_agent.video_rubric import VideoRubric

    rubric_csv = ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"
    known = pd.read_csv(rubric_csv).iloc[0].to_dict()
    flags = ("tem_rosto", "pessoa_fala_para_camera", "close_up_produto", "before_after_ou_transformacao",
             "try_on_ou_demonstracao", "texto_na_tela", "tem_musica", "chamada_para_acao")
    answer = {k: (bool(known[k]) if k in flags else known[k]) for k in known if k not in
              ("video_id", "gemini_run_id", "model", "prompt_version", "prompt_tokens")}
    answer["elementos_video_top5"] = str(answer["elementos_video_top5"]).split(" | ")
    answer["elementos_audio_top5"] = str(answer["elementos_audio_top5"]).split(" | ")
    answer = {k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in answer.items()}
    payload = {"candidates": [{"content": {"parts": [{"text": json.dumps(answer)}]}}]}
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=0):
        calls.append(json.loads(req.data)["contents"][0]["parts"][0]["file_data"]["file_uri"])
        return Resp(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "teste")
    vr = VideoRubric(tmp_path, live=True)  # raiz vazia: nada em cache
    vr.root = tmp_path
    vr._load_collector = lambda: VideoRubric(ROOT)._load_collector()
    rubric, source = vr.get("AAAAAAAAAAA", "moda")
    assert source == "gemini_ao_vivo" and calls == ["https://www.youtube.com/watch?v=AAAAAAAAAAA"]
    assert rubric["tem_rosto"] in (0, 1) and isinstance(rubric["elementos_video_top5"], str)
    assert vr.get("AAAAAAAAAAA", "moda")[1] == "cache" and len(calls) == 1  # segunda vez: cache, sem nova chamada

    p = predictor
    sil = silver.set_index("video_id")
    row = next(gold.sample(1, random_state=7).itertuples())
    out = p.predict(as_input(row, sil.loc[row.video_id]), attributes=(rubric, source))
    assert out.model_variant == "final" and out.rubric_source == "gemini_ao_vivo"
