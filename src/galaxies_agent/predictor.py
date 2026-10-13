"""Funcao de previsao da tool: criativo + contexto -> previsao, intervalo, drivers e comparaveis.

O numero vem SEMPRE do modelo treinado (CatBoost do E-08). O texto de `explanation` e montado de forma
deterministica a partir do SHAP; um LLM/agente pode reescrever, mas nao alterar os numeros.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .embedder import ThumbnailEmbedder, youtube_id
from .schemas import Comparable, CreativeInput, Driver, PredictionOutput, RankedCreative, RankingOutput

ROOT = Path(os.environ.get("GALAXIES_ROOT", Path(__file__).resolve().parents[2]))
LOCAL_TZ = ZoneInfo("America/Sao_Paulo")
MIN_HISTORY = 30          # mesmo criterio de entrada de canal do piloto
MAX_GAP_DAYS = 60         # acima disso o intervalo desde o ultimo post sai do padrao de treino
BANDS = [(-0.30, "abaixo"), (0.30, "na_media"), (1.00, "acima")]  # limites do desvio em log

LABELS = {
    "thumb_embedding": "estilo visual da thumbnail (embedding SigLIP 2)",
    "baseline_log_likes": "historico de curtidas do canal",
    "duration_seconds": "duracao do Short",
    "channel_prior_posts": "quantidade de Shorts anteriores do canal",
    "days_since_previous_post": "dias desde o Short anterior",
    "title_length": "tamanho do titulo",
    "description_length": "tamanho da descricao",
    "tag_count": "quantidade de tags",
    "channel_id": "canal",
    "sector": "setor",
    "channel_type": "tipo de canal (marca/criador)",
    "publish_hour_utc": "horario de publicacao (UTC)",
    "publish_weekday": "dia da semana",
}
WEEKDAYS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]


class Predictor:
    def __init__(self, model_dir: Optional[Path] = None, extra_embedding_cache: Optional[list[Path]] = None):
        if model_dir is None:
            latest = json.loads((ROOT / "artifacts" / "tool_model" / "LATEST.json").read_text(encoding="utf-8"))
            model_dir = ROOT / latest["path"]
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))
        self.version = self.meta["model_version"]
        self.numeric = self.meta["numeric_features"]
        self.categorical = self.meta["categorical_features"]
        self.emb_cols = self.meta["embedding_features"]
        self.model_emb = CatBoostRegressor().load_model(str(self.dir / "model_with_embedding.cbm"))
        self.model_base = CatBoostRegressor().load_model(str(self.dir / "model_without_embedding.cbm"))
        pca = np.load(self.dir / "pca.npz")
        self.pca_mean, self.pca_components = pca["mean"], pca["components"]
        self.history = pd.read_csv(self.dir / "channel_history.csv", parse_dates=["published_at"])
        self.channels = pd.read_csv(self.dir / "channels.csv").set_index("channel_id")
        self.comparables = pd.read_csv(self.dir / "comparables.csv")
        self.comp_emb = np.load(self.dir / "comparables_embeddings.npy")
        cache = [ROOT / self.meta["inputs"]["embeddings"]["path"]] + list(extra_embedding_cache or [])
        self.embedder = ThumbnailEmbedder(self.meta["embedding"]["model"], cache)
        self.calibration = self.meta["interval"]["calibration"]

    # ------------------------------------------------------------ features
    def _planned_utc(self, when: datetime) -> datetime:
        if when.tzinfo is None:
            when = when.replace(tzinfo=LOCAL_TZ)
        return when.astimezone(timezone.utc)

    def channel_context(self, item: CreativeInput, planned: datetime) -> dict:
        if item.channel_stats is not None:
            stats = item.channel_stats
            likes = list(stats.prior_likes)
            last = self._planned_utc(stats.last_post_at) if stats.last_post_at else None
            return {"channel_id": item.channel_id or f"externo:{stats.channel_name or 'sem_nome'}",
                    "channel_name": stats.channel_name or item.channel_id or "canal externo",
                    "channel_type": stats.channel_type, "prior_likes": likes, "last_post_at": last,
                    "median_duration": stats.median_duration_seconds, "in_training": False, "source": "channel_stats"}
        rows = self.history[(self.history["channel_id"] == item.channel_id)
                            & (self.history["published_at"] < pd.Timestamp(planned))].sort_values("published_at")
        if rows.empty:
            raise ValueError(f"canal {item.channel_id} sem historico anterior a {planned.isoformat()}; envie channel_stats")
        info = self.channels.loc[item.channel_id] if item.channel_id in self.channels.index else None
        return {"channel_id": item.channel_id, "channel_name": rows["channel_name"].iat[-1],
                "channel_type": rows["channel_type"].iat[-1], "prior_likes": rows["like_count"].astype(int).tolist(),
                "last_post_at": rows["published_at"].iat[-1].to_pydatetime(),
                "median_duration": float(rows["duration_seconds"].median()),
                "in_training": bool(info["in_training"]) if info is not None else False, "source": "base_silver"}

    def build_row(self, item: CreativeInput) -> tuple[dict, dict]:
        planned = self._planned_utc(item.planned_datetime)
        ctx = self.channel_context(item, planned)
        likes = np.asarray(ctx["prior_likes"], dtype=float)
        gap = (planned - ctx["last_post_at"]).total_seconds() / 86400 if ctx["last_post_at"] else np.nan
        row = {
            "duration_seconds": float(item.duration_seconds or ctx["median_duration"] or 30.0),
            "channel_prior_posts": float(len(likes)),
            "baseline_log_likes": float(np.log1p(np.median(likes))),
            "days_since_previous_post": float(gap),
            "title_length": float(len(item.caption)),
            "description_length": float(len(item.description)),
            "tag_count": float(len(item.tags)),
            "channel_id": ctx["channel_id"], "sector": item.sector, "channel_type": ctx["channel_type"],
            "publish_hour_utc": str(planned.hour), "publish_weekday": str(planned.weekday()),
        }
        return row, ctx

    # ------------------------------------------------------------ previsao
    def _pool(self, row: dict, use_emb: bool, emb_pcs: Optional[np.ndarray]) -> Pool:
        numeric = self.numeric + (self.emb_cols if use_emb else [])
        values = {c: [row[c]] for c in self.numeric + self.categorical}
        if use_emb:
            values.update({c: [float(v)] for c, v in zip(self.emb_cols, emb_pcs)})
        frame = pd.DataFrame(values)[numeric + self.categorical]
        for c in self.categorical:
            frame[c] = frame[c].astype(str)
        return Pool(frame, cat_features=self.categorical)

    def _drivers(self, model: CatBoostRegressor, pool: Pool, row: dict, use_emb: bool) -> list[Driver]:
        shap = model.get_feature_importance(pool, type="ShapValues")[0][:-1]
        names = list(pool.get_feature_names())
        grouped: dict[str, float] = {}
        for name, value in zip(names, shap):
            key = "thumb_embedding" if name.startswith("emb_pc") else name
            grouped[key] = grouped.get(key, 0.0) + float(value)
        out = []
        for key, value in sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:5]:
            if abs(value) < 1e-4:
                continue
            out.append(Driver(feature=key, label=LABELS.get(key, key), value=self._display(key, row),
                              effect_log=round(value, 4), direction="aumenta" if value > 0 else "reduz"))
        return out

    @staticmethod
    def _display(key: str, row: dict) -> str:
        if key == "thumb_embedding":
            return "imagem do criativo"
        if key == "baseline_log_likes":
            return f"mediana de {np.expm1(row[key]):,.0f} curtidas".replace(",", ".")
        if key == "publish_weekday":
            return WEEKDAYS[int(row[key])]
        if key == "publish_hour_utc":
            return f"{row[key]}h UTC"
        value = row.get(key)
        return f"{value:.0f}" if isinstance(value, float) and not np.isnan(value) else str(value)

    def _comparables(self, vector: Optional[np.ndarray], ctx: dict, sector: str, k: int = 3,
                     exclude: Optional[str] = None) -> list[Comparable]:
        if vector is None:
            return []
        sims = self.comp_emb @ (vector / np.linalg.norm(vector))
        frame = self.comparables.assign(similarity=sims)
        frame = frame[frame["video_id"] != exclude]  # o proprio criativo nao e evidencia de si mesmo
        same = frame[frame["channel_id"] == ctx["channel_id"]]
        pool = same if len(same) >= k else frame[frame["sector"] == sector]
        top = pool.nlargest(k, "similarity")
        return [Comparable(video_id=r.video_id, title=str(r.title)[:120], channel_name=r.channel_name,
                           published_at=str(r.published_at)[:10], likes=int(r.likes),
                           relative_to_channel=round(float(r.relative_to_channel), 2), similarity=round(float(r.similarity), 3),
                           url=f"https://www.youtube.com/shorts/{r.video_id}") for r in top.itertuples()]

    def predict(self, item: CreativeInput) -> PredictionOutput:
        row, ctx = self.build_row(item)
        vector, emb_source = self.embedder.embed(item.creative)
        use_emb = vector is not None
        emb_pcs = (vector - self.pca_mean) @ self.pca_components.T if use_emb else None
        model = self.model_emb if use_emb else self.model_base
        pool = self._pool(row, use_emb, emb_pcs)
        residual = float(model.predict(pool)[0])
        baseline = row["baseline_log_likes"]
        pred_log = baseline + residual

        cal = self.calibration["with_embedding" if use_emb else "without_embedding"]
        lo_q, hi_q = cal.get(ctx["channel_type"], cal["global"])
        likes = int(round(np.expm1(pred_log)))
        low, high = int(round(max(0.0, np.expm1(pred_log + lo_q)))), int(round(np.expm1(pred_log + hi_q)))

        reasons = []
        if row["channel_prior_posts"] < MIN_HISTORY:
            reasons.append(f"historico curto: {int(row['channel_prior_posts'])} Shorts anteriores (minimo {MIN_HISTORY})")
        if not ctx["in_training"]:
            reasons.append("canal fora da base de treino: o efeito proprio do canal nao foi aprendido")
        if not use_emb:
            reasons.append(f"sem embedding da thumbnail ({emb_source}); usado modelo reserva sem conteudo visual")
        if np.isnan(row["days_since_previous_post"]) or row["days_since_previous_post"] > MAX_GAP_DAYS:
            reasons.append("intervalo desde o ultimo Short fora do padrao de treino (> 60 dias ou desconhecido)")
        band = next((name for limit, name in BANDS if residual < limit), "viral")
        drivers = self._drivers(model, pool, row, use_emb)
        comps = self._comparables(vector, ctx, item.sector, exclude=youtube_id(item.creative))
        baseline_likes = int(round(np.expm1(baseline)))
        relative = float(np.exp(residual))

        def br(n: float, decimals: int = 0) -> str:
            text = f"{n:,.{decimals}f}"
            return text.replace(",", "_").replace(".", ",").replace("_", ".")

        explanation = (
            f"Previsao de {br(likes)} curtidas (intervalo de 80%: {br(low)} a {br(high)}) para {ctx['channel_name']}, "
            f"{br(relative, 2)}x a mediana historica do canal ({br(baseline_likes)}). "
            + ("Principais fatores: " + "; ".join(f"{d.label} {d.direction} ({d.effect_log:+.2f} em log)" for d in drivers[:3]) + ". " if drivers else "")
            + (f"Confianca baixa: {'; '.join(reasons)}." if reasons else "Confianca alta: canal com historico e thumbnail processada.")
        )

        features_used = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        features_used["embedding"] = emb_source
        return PredictionOutput(
            creative_id=item.creative_id or youtube_id(item.creative) or item.creative, predicted_metric=likes,
            interval_low=low, interval_high=high, interval_level=self.meta["interval"]["nominal"],
            confidence_flag="baixa" if reasons else "alta", confidence_reasons=reasons,
            channel_baseline_likes=baseline_likes, relative_to_channel=round(relative, 3), performance_band=band,
            top_drivers=drivers, explanation=explanation, comparables=comps, features_used=features_used,
            model_version=self.version, embedding_source=emb_source,
        )

    def rank(self, items: list[CreativeInput]) -> RankingOutput:
        predictions = [self.predict(item) for item in items]
        order = sorted(range(len(predictions)), key=lambda i: -predictions[i].relative_to_channel)
        ranked = [RankedCreative(rank=pos + 1, **predictions[i].model_dump()) for pos, i in enumerate(order)]
        note = ("Ordenado pelo desempenho previsto relativo a mediana de cada canal. Diferencas menores que a "
                "largura do intervalo nao sao decisivas; no rolling-origin, o modelo acerta ~57% dos pares "
                "dentro do mesmo canal (moeda = 50%).")
        return RankingOutput(ranking=ranked, note=note)

    def channel_summary(self, channel_id: str, last: int = 10) -> dict:
        rows = self.history[self.history["channel_id"] == channel_id].sort_values("published_at")
        if rows.empty:
            raise ValueError(f"canal {channel_id} nao esta na base")
        recent = rows.tail(last)
        return {"channel_id": channel_id, "channel_name": rows["channel_name"].iat[0], "sector": rows["sector"].iat[0],
                "channel_type": rows["channel_type"].iat[0], "shorts_na_base": int(len(rows)),
                "mediana_curtidas": float(rows["like_count"].median()),
                "periodo": [str(rows["published_at"].min())[:10], str(rows["published_at"].max())[:10]],
                "ultimos": [{"video_id": r.video_id, "titulo": str(r.title)[:100], "publicado": str(r.published_at)[:10],
                             "curtidas": int(r.like_count)} for r in recent.itertuples()]}

    def model_info(self) -> dict:
        keep = ("model_version", "reference_experiment", "dataset_version", "training_rows", "training_channels",
                "test_split", "target", "embedding", "interval", "baseline_rule")
        return {k: self.meta[k] for k in keep}
