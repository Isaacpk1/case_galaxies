"""E-10 (PCA, texto, combinacao) e E-11 (canal nao visto), protocolo pre-registrado em docs/07.

Retomavel: cada tarefa grava suas previsoes em artifacts/feature_experiments/e10_e11/tasks/<tarefa>.csv e
e pulada se ja existir. Rode varias vezes ate terminar e depois --report.

    python scripts/run_e10_e11_experiments.py --max-seconds 160     # executa tarefas pendentes
    python scripts/run_e10_e11_experiments.py --status
    python scripts/run_e10_e11_experiments.py --report
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "feature_experiments" / "e10_e11"
TASKS = OUT / "tasks"
E08 = ROOT / "artifacts" / "feature_experiments" / "e08_catboost_residual"
THUMB_EMB = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"
TEXT_DIR = ROOT / "data" / "silver" / "features" / "text_mpnet_v1"
TARGET, BASELINE = "target_log_likes", "baseline_log_likes"
BASE_NUM = ["duration_seconds", "channel_prior_posts", "baseline_log_likes", "days_since_previous_post",
            "title_length", "description_length", "tag_count"]
BASE_CAT = ["channel_id", "sector", "channel_type", "publish_hour_utc", "publish_weekday"]
THUMB_NUM = ["thumb_is_vertical_pillarbox", "thumb_letterbox_fraction", "thumb_is_low_resolution",
             "thumb_brightness_mean", "thumb_brightness_std", "thumb_saturation_mean", "thumb_colorfulness",
             "thumb_dark_pixel_pct", "thumb_light_pixel_pct", "thumb_edge_density", "thumb_has_text",
             "thumb_ocr_log_word_count", "thumb_ocr_text_area_pct", "thumb_ocr_uppercase_ratio",
             "thumb_ocr_has_digits", "thumb_ocr_has_question", "thumb_ocr_has_exclamation"]
THUMB_CAT = ["thumb_dominant_hue"]
TEXT_NUM, TEXT_CAT = ["has_cta", "title_tag_overlap"], ["duration_bucket"]
FRACTIONS = (0.50, 0.65, 0.80)
SEED = 42

# arm: (numericas, categoricas, dims thumb, texto)
ARMS = {
    "CB_EMB_pca8": (BASE_NUM, BASE_CAT, 8, None),
    "CB_EMB_pca16": (BASE_NUM, BASE_CAT, 16, None),
    "CB_EMB_pca32": (BASE_NUM, BASE_CAT, 32, None),
    "CB_FULL_fixed": (BASE_NUM + THUMB_NUM + TEXT_NUM, BASE_CAT + THUMB_CAT + TEXT_CAT, 16, None),
    "CB_EMB_text_title": (BASE_NUM, BASE_CAT, 16, "title"),
    "CB_EMB_text_title_desc": (BASE_NUM, BASE_CAT, 16, "title_desc"),
}


def params() -> dict:
    final = json.loads((E08 / "run_info.json").read_text())["final_params"]
    return {k: (int(v) if k in ("depth", "iterations") else v) for k, v in final.items()}


def load() -> tuple[pd.DataFrame, dict, dict]:
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text())
    gold = pd.read_csv(ROOT / latest["path"], parse_dates=["published_at"])
    data = gold[gold["split"] == "train"].reset_index(drop=True)
    data["thumb_ocr_log_word_count"] = np.log1p(data["thumb_ocr_word_count"].fillna(0))
    z = np.load(THUMB_EMB, allow_pickle=True)
    thumb = dict(zip(z["video_id"].astype(str), z["embeddings"].astype(np.float32)))
    text = {}
    runs = sorted(p for p in TEXT_DIR.glob("*") if (p / "text_embeddings.npz").exists()) if TEXT_DIR.exists() else []
    if runs:
        t = np.load(runs[-1] / "text_embeddings.npz", allow_pickle=True)
        for kind in ("title", "title_desc"):
            text[kind] = dict(zip(t["video_id"].astype(str), t[kind].astype(np.float32)))
    return data, thumb, text


def folds(frame: pd.DataFrame):
    for pos, frac in enumerate(FRACTIONS):
        nxt = FRACTIONS[pos + 1] if pos + 1 < len(FRACTIONS) else 1.0
        fit, val = [], []
        for _, ch in frame.sort_values(["channel_id", "published_at", "video_id"]).groupby("channel_id"):
            end = max(5, int(np.floor(len(ch) * frac)))
            stop = min(max(end + 1, int(np.floor(len(ch) * nxt))), len(ch))
            fit.extend(ch.index[:end])
            val.extend(ch.index[end:stop])
        yield f"origin_{pos + 1}", fit, val


def pca_fit(matrix: np.ndarray, dim: int):
    mean = matrix.mean(0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    return mean, vt[:dim]


def add_pcs(fit: pd.DataFrame, val: pd.DataFrame, emb: dict, dim: int, prefix: str) -> list[str]:
    a = np.vstack([emb[v] for v in fit["video_id"]])
    b = np.vstack([emb[v] for v in val["video_id"]])
    mean, comp = pca_fit(a, dim)
    cols = [f"{prefix}{i:02d}" for i in range(dim)]
    fit[cols] = (a - mean) @ comp.T
    val[cols] = (b - mean) @ comp.T
    return cols


def fit_predict(fit, val, num, cat) -> np.ndarray:
    def pool(frame, label):
        x = frame[num + cat].copy()
        for c in cat:
            x[c] = x[c].fillna("<missing>").astype(str)
        return Pool(x, (frame[TARGET] - frame[BASELINE]).to_numpy() if label else None, cat_features=cat)
    model = CatBoostRegressor(loss_function="MAE", random_seed=SEED, verbose=False, allow_writing_files=False,
                              thread_count=-1, **params())
    model.fit(pool(fit, True))
    return val[BASELINE].to_numpy() + model.predict(pool(val, False))


def task_list(data: pd.DataFrame, text: dict) -> list[tuple]:
    tasks = []
    for arm, (_, _, _, txt) in ARMS.items():
        if txt and txt not in text:
            continue
        for origin in ("origin_1", "origin_2", "origin_3"):
            tasks.append(("E10", arm, origin, None))
    for channel in sorted(data["channel_id"].unique()):
        for origin in ("origin_1", "origin_2", "origin_3"):
            tasks.append(("E11", "LOCO", origin, channel))
    return tasks


def task_name(task) -> str:
    exp, arm, origin, channel = task
    return f"{exp}__{arm}__{origin}" + (f"__{channel}" if channel else "")


def run_task(task, data, thumb, text) -> pd.DataFrame:
    exp, arm, origin, channel = task
    fit_idx, val_idx = next((f, v) for o, f, v in folds(data) if o == origin)
    fit, val = data.loc[fit_idx].copy(), data.loc[val_idx].copy()
    keep = ["video_id", "channel_id", "channel_name", "channel_type", "sector", TARGET, BASELINE]
    if exp == "E10":
        num, cat, dim, txt = ARMS[arm]
        cols = add_pcs(fit, val, thumb, dim, "emb_")
        if txt:
            cols += add_pcs(fit, val, text[txt], 16, "txt_")
        out = val[keep].copy()
        out["pred"] = fit_predict(fit, val, num + cols, cat)
        out["model"], out["origin"] = arm, origin
        return out
    # E11: deixa o canal fora do treino; preve a janela de validacao dele
    fit, val = fit[fit["channel_id"] != channel].copy(), val[val["channel_id"] == channel].copy()
    cat = [c for c in BASE_CAT if c != "channel_id"]
    frames = []
    b0 = val[keep].copy(); b0["pred"] = val[BASELINE].to_numpy(); b0["model"] = "B0"; frames.append(b0)
    base = val[keep].copy(); base["pred"] = fit_predict(fit, val, BASE_NUM, cat); base["model"] = "CB_BASE_loco"; frames.append(base)
    cols = add_pcs(fit, val, thumb, 16, "emb_")
    emb = val[keep].copy(); emb["pred"] = fit_predict(fit, val, BASE_NUM + cols, cat); emb["model"] = "CB_EMB_loco"; frames.append(emb)
    out = pd.concat(frames)
    out["origin"] = origin
    return out


def pairwise(actual, pred) -> float:
    left, right = np.triu_indices(len(actual), k=1)
    a, p = np.sign(actual[left] - actual[right]), np.sign(pred[left] - pred[right])
    ok = a != 0
    return float(np.where(p[ok] == 0, 0.5, (a[ok] == p[ok]).astype(float)).mean()) if ok.any() else np.nan


def per_channel(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, origin, channel), g in pred.groupby(["model", "origin", "channel_name"]):
        a, p = g[TARGET].to_numpy(), g["pred"].to_numpy()
        rows.append({"model": model, "origin": origin, "channel_name": channel, "channel_type": g["channel_type"].iat[0],
                     "mae": float(np.mean(np.abs(a - p))), "pair_fair": pairwise(a, p)})
    return pd.DataFrame(rows).groupby(["model", "channel_name", "channel_type"], as_index=False)[["mae", "pair_fair"]].mean()


def paired(pc: pd.DataFrame, cand: str, ref: str, metric: str) -> dict:
    v = pc.pivot(index="channel_name", columns="model", values=metric)
    d = (v[cand] - v[ref]).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d, (10000, len(d))).mean(1)
    signs = rng.choice((-1, 1), (10000, len(d)))
    better = (d < 0) if metric == "mae" else (d > 0)
    return {"comparacao": f"{cand} - {ref}", "metrica": metric, "delta": round(float(d.mean()), 4),
            "ic95": [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
            "p": round(float((np.abs((signs * d).mean(1)) >= abs(d.mean())).mean()), 4),
            "canais_melhor": int(better.sum()), "n": int(len(d))}


def report() -> None:
    files = sorted(TASKS.glob("*.csv"))
    pred = pd.concat([pd.read_csv(f) for f in files])
    e08 = pd.read_csv(E08 / "rolling_origin_predictions.csv")
    b0 = e08[["video_id", "channel_id", "channel_name", "channel_type", "sector", TARGET, BASELINE, "origin", "pred_B0"]].rename(columns={"pred_B0": "pred"}).assign(model="B0")
    result = {}
    e10 = pd.concat([pred[pred["model"].isin(ARMS)], b0])
    if len(e10):
        pc = per_channel(e10)
        result["E10_resumo"] = pc.groupby("model")[["mae", "pair_fair"]].mean().round(4).reset_index().to_dict("records")
        ref = "CB_EMB_pca16"
        result["E10_comparacoes"] = [paired(pc, m, ref, metric) for m in pc["model"].unique() if m not in (ref, "B0")
                                     for metric in ("mae", "pair_fair")] + [paired(pc, ref, "B0", "mae")]
    e11 = pred[pred["model"].str.endswith("_loco") | (pred["model"] == "B0")]
    if len(e11):
        pc = per_channel(e11)
        result["E11_resumo"] = pc.groupby("model")[["mae", "pair_fair"]].mean().round(4).reset_index().to_dict("records")
        result["E11_comparacoes"] = [paired(pc, "CB_EMB_loco", "B0", m) for m in ("mae", "pair_fair")] + \
            [paired(pc, "CB_EMB_loco", "CB_BASE_loco", m) for m in ("mae", "pair_fair")] + [paired(pc, "CB_BASE_loco", "B0", "mae")]
        result["E11_por_tipo"] = pc.groupby(["model", "channel_type"])[["mae", "pair_fair"]].mean().round(4).reset_index().to_dict("records")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=160)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", choices=["E10", "E11"])
    args = parser.parse_args()
    if args.report:
        report(); return
    data, thumb, text = load()
    TASKS.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in task_list(data, text) if not args.only or t[0] == args.only]
    pending = [t for t in tasks if not (TASKS / f"{task_name(t)}.csv").exists()]
    print(f"tarefas: {len(tasks)} | pendentes: {len(pending)} | texto disponivel: {sorted(text)}")
    if args.status:
        return
    start = time.time()
    for task in pending:
        if time.time() - start > args.max_seconds:
            break
        t0 = time.time()
        out = run_task(task, data, thumb, text)
        tmp = TASKS / f"{task_name(task)}.csv.tmp"
        out.to_csv(tmp, index=False)
        tmp.replace(TASKS / f"{task_name(task)}.csv")
        print(f"ok {task_name(task)} ({time.time() - t0:.0f}s)", flush=True)
    left = len([t for t in tasks if not (TASKS / f"{task_name(t)}.csv").exists()])
    print(f"restantes: {left}")


if __name__ == "__main__":
    main()
