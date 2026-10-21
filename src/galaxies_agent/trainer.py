"""Treino dinamico (extensao do TAP, secao 4.2): base propria de um cliente -> modelo especialista avaliado.

Recebe um CSV com o esquema minimo do TAP (secao 7) e repete o fluxo do projeto em escala pequena:
validacao das entradas, features so com posts anteriores, alvo residual sobre a mediana do canal (B0),
CatBoost com perda MAE (hiperparametros do CB_BASE do E-08), avaliacao temporal nos 20% mais recentes de
cada canal contra o B0 e intervalo de 80% calibrado num holdout interno do treino.

Esquema minimo (colunas obrigatorias): post_id, channel_id, published_at, caption, metric_target, sector.
Opcionais: channel_type (brand|creator), duration_seconds, description, tags (lista JSON ou separada por virgula),
media_ref, channel_stats_at_collection. Colunas extras sao ignoradas.

So metadados e historico entram (a variante sem imagem); thumbnail e rubrica do video exigem o pipeline
multimodal e ficam para quando o cliente liberar a midia. O relatorio diz se o modelo bate o B0.

    from galaxies_agent.trainer import train_client_model
    report = train_client_model("base_cliente.csv", nome="cliente_x")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pydantic import BaseModel, Field, field_validator

from .predictor import ROOT

REQUIRED = ["post_id", "channel_id", "published_at", "caption", "metric_target", "sector"]
NUMERIC = ["duration_seconds", "channel_prior_posts", "baseline_log_likes", "days_since_previous_post",
           "title_length", "description_length", "tag_count"]
CATEGORICAL = ["channel_id", "sector", "channel_type", "publish_hour_utc", "publish_weekday"]
PARAMS = {"depth": 8, "learning_rate": 0.1, "l2_leaf_reg": 10, "random_strength": 1.0, "bagging_temperature": 0.5,
          "iterations": 259}  # CB_BASE do E-08 (mesmos do pacote da tool)
OUT_BASE = ROOT / "artifacts" / "client_models"


class ClientRow(BaseModel):
    post_id: str
    channel_id: str
    published_at: datetime
    caption: str = ""
    metric_target: float = Field(..., ge=0)
    sector: str
    channel_type: Literal["brand", "creator"] = "brand"
    duration_seconds: Optional[float] = Field(None, gt=0)

    @field_validator("caption", mode="before")
    @classmethod
    def empty_caption(cls, value):
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


class ClientTrainingReport(BaseModel):
    nome: str
    modelo_dir: str
    linhas_validas: int
    linhas_descartadas: int
    canais: int
    treino: int
    teste: int
    periodo: list[str]
    metricas_teste: dict
    comparacao_vs_b0: dict
    intervalo_80: dict
    decisao: str
    avisos: list[str]


def _tags(value) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return 0
    try:
        parsed = json.loads(value)
        return len(parsed) if isinstance(parsed, list) else 1
    except (TypeError, ValueError):
        return len([t for t in str(value).split(",") if t.strip()])


def load_client_base(source: Union[str, Path, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"faltam colunas do esquema minimo do TAP: {missing}")
    valid, dropped = [], 0
    for rec in raw.to_dict("records"):
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        try:
            row = ClientRow(**{k: clean.get(k) for k in ClientRow.model_fields if clean.get(k) is not None})
        except Exception:
            dropped += 1
            continue
        valid.append({**row.model_dump(), "description": clean.get("description") or "", "tags": clean.get("tags")})
    data = pd.DataFrame(valid).drop_duplicates("post_id")
    if data.empty:
        raise ValueError("nenhuma linha valida no arquivo")
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    return data, dropped


def build_features(data: pd.DataFrame, min_prior_posts: int = 5) -> pd.DataFrame:
    """Historico so com posts ESTRITAMENTE anteriores (posts no mesmo instante nao se enxergam)."""
    data = data.sort_values(["channel_id", "published_at", "post_id"]).reset_index(drop=True).copy()
    data["channel_prior_posts"] = np.nan
    data["channel_prior_median"] = np.nan
    data["days_since_previous_post"] = np.nan
    for _, idx in data.groupby("channel_id", sort=False).groups.items():
        ch = data.loc[idx]
        t = ch["published_at"].to_numpy()
        y = ch["metric_target"].to_numpy(dtype=float)
        for pos, i in enumerate(idx):
            prior = t < t[pos]
            n = int(prior.sum())
            data.at[i, "channel_prior_posts"] = n
            if n:
                data.at[i, "channel_prior_median"] = float(np.median(y[prior]))
                data.at[i, "days_since_previous_post"] = (t[pos] - t[prior].max()) / np.timedelta64(1, "D")
    data["baseline_log_likes"] = np.log1p(data["channel_prior_median"])
    data["target_log_likes"] = np.log1p(data["metric_target"])
    data["publish_hour_utc"] = data["published_at"].dt.hour.astype(str)
    data["publish_weekday"] = data["published_at"].dt.dayofweek.astype(str)
    data["title_length"] = data["caption"].fillna("").str.len().astype(float)
    data["description_length"] = data["description"].fillna("").astype(str).str.len().astype(float)
    data["tag_count"] = data["tags"].map(_tags).astype(float)
    data["duration_seconds"] = pd.to_numeric(data["duration_seconds"], errors="coerce")
    return data[data["channel_prior_posts"].ge(min_prior_posts)].copy()


def temporal_split(data: pd.DataFrame, fraction: float) -> pd.Series:
    split = pd.Series("train", index=data.index)
    for _, ch in data.sort_values(["published_at", "post_id"]).groupby("channel_id"):
        size = int(np.ceil(len(ch) * fraction))
        start = ch.iloc[-size]["published_at"]
        split.loc[ch.index[ch["published_at"].ge(start)]] = "test"
    return split


def _pool(frame: pd.DataFrame, label: bool) -> Pool:
    x = frame[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna("<missing>").astype(str)
    return Pool(x, (frame["target_log_likes"] - frame["baseline_log_likes"]).to_numpy() if label else None, cat_features=CATEGORICAL)


def _fit(frame: pd.DataFrame) -> CatBoostRegressor:
    return CatBoostRegressor(loss_function="MAE", random_seed=42, verbose=False, allow_writing_files=False,
                             thread_count=-1, **PARAMS).fit(_pool(frame, True))


def _pairwise(a: np.ndarray, p: np.ndarray) -> float:
    l, r = np.triu_indices(len(a), k=1)
    sa, sp = np.sign(a[l] - a[r]), np.sign(p[l] - p[r])
    ok = sa != 0
    return float(np.where(sp[ok] == 0, 0.5, (sa[ok] == sp[ok]).astype(float)).mean()) if ok.any() else float("nan")


def _per_channel(test: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for ch, g in test.groupby("channel_id"):
        a, p = g["target_log_likes"].to_numpy(), g[col].to_numpy()
        rows.append({"channel_id": ch, "mae": float(np.mean(np.abs(a - p))), "pares": _pairwise(a, p) if len(g) > 1 else np.nan})
    return pd.DataFrame(rows).set_index("channel_id")


def _paired(diff: pd.Series, lower_is_better: bool) -> dict:
    d = diff.dropna().to_numpy()
    if len(d) < 2:
        return {"delta": round(float(d.mean()), 4) if len(d) else None, "p": None, "canais_melhor": int((d < 0).sum() if lower_is_better else (d > 0).sum()), "n": int(len(d))}
    s = np.random.default_rng(42).choice((-1, 1), (10000, len(d)))
    return {"delta": round(float(d.mean()), 4), "p": round(float((np.abs((s * d).mean(1)) >= abs(d.mean())).mean()), 4),
            "canais_melhor": int((d < 0).sum() if lower_is_better else (d > 0).sum()), "n": int(len(d))}


def train_client_model(source: Union[str, Path, pd.DataFrame], nome: str, test_fraction: float = 0.2,
                       min_prior_posts: int = 5, out_base: Optional[Path] = None) -> ClientTrainingReport:
    data, dropped = load_client_base(source)
    feats = build_features(data, min_prior_posts)
    warnings = []
    if feats["channel_id"].nunique() < 3:
        warnings.append("menos de 3 canais: a comparacao pareada por canal tem pouco poder")
    if len(feats) < 300:
        warnings.append(f"so {len(feats)} linhas com >= {min_prior_posts} posts anteriores: resultado instavel")
    if feats["duration_seconds"].isna().all():
        warnings.append("sem duration_seconds: o modelo usa apenas historico, texto e horario")
    feats["split"] = temporal_split(feats, test_fraction)
    train, test = feats[feats["split"] == "train"].copy(), feats[feats["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("base pequena demais para separar treino e teste por canal")

    # intervalo: calibra nos 20% mais recentes do treino com um modelo que nao os viu
    inner = temporal_split(train, test_fraction)
    cal_model = _fit(train[inner == "train"])
    cal = train[inner == "test"]
    err = cal["target_log_likes"] - (cal["baseline_log_likes"] + cal_model.predict(_pool(cal, False)))
    lo_q, hi_q = (float(q) for q in np.quantile(err, [0.1, 0.9]))

    model = _fit(train)
    test["pred_B0"] = test["baseline_log_likes"]
    test["pred_modelo"] = test["baseline_log_likes"] + model.predict(_pool(test, False))
    inside = (test["target_log_likes"] >= test["pred_modelo"] + lo_q) & (test["target_log_likes"] <= test["pred_modelo"] + hi_q)
    pm, pb = _per_channel(test, "pred_modelo"), _per_channel(test, "pred_B0")
    metrics = {name: {"mae_macro_canal": round(float(pc["mae"].mean()), 4), "pares_macro_canal": round(float(pc["pares"].mean()), 4)}
               for name, pc in (("B0", pb), ("modelo", pm))}
    comp = {"mae": _paired(pm["mae"] - pb["mae"], True), "pares": _paired(pm["pares"] - pb["pares"], False)}
    beats = comp["mae"]["delta"] is not None and comp["mae"]["delta"] < 0 and (comp["mae"]["p"] or 1) < 0.05
    decision = ("usar o modelo especialista: bate a mediana do canal em MAE com p < 0,05" if beats else
                "usar com cautela: o modelo nao bate a mediana do canal com significancia nesta base; prefira a mediana do canal para o numero e o modelo so para ordenar"
                if metrics["modelo"]["pares_macro_canal"] > 0.5 else
                "nao usar: o modelo nao supera a mediana do canal nem ordena acima do acaso nesta base")

    final = _fit(feats)  # pacote final com todos os dados, depois da avaliacao
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (out_base or OUT_BASE) / f"{nome}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    final.save_model(str(out / "model.cbm"))
    feats[["post_id", "channel_id", "published_at", "metric_target", "channel_type", "sector", "duration_seconds"]].to_csv(out / "history.csv", index=False)
    report = ClientTrainingReport(
        nome=nome, modelo_dir=str(out), linhas_validas=int(len(data)), linhas_descartadas=int(dropped),
        canais=int(feats["channel_id"].nunique()), treino=int(len(train)), teste=int(len(test)),
        periodo=[str(feats["published_at"].min()), str(feats["published_at"].max())],
        metricas_teste=metrics, comparacao_vs_b0=comp,
        intervalo_80={"quantis_log": [round(lo_q, 4), round(hi_q, 4)], "cobertura_teste": round(float(inside.mean()), 4)},
        decisao=decision, avisos=warnings)
    (out / "metadata.json").write_text(json.dumps({"report": report.model_dump(), "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
                                                   "params": PARAMS, "target": "log1p(metric_target) - log1p(mediana dos posts anteriores do canal)"},
                                                  indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
