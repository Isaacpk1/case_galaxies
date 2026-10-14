"""Build the four versioned feature groups (A contexto, B texto, C thumbnail, D transcricao).

Every group has one row per eligible Short of the Silver run, so a missing
modality is a coverage flag (has_thumbnail / has_transcript = 0) and never
drops the Short.  Only information available before publication is used:
the channel history aggregates look strictly at earlier posts, and the
video's own likes/views/comments never become features.

Usage:
  python scripts/pipeline/build_feature_store.py --silver-run-id 20261002T003000Z
  python scripts/pipeline/build_feature_store.py --silver-run-id ... --groups context text
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store import ROOT, write_feature_set
from prepare_pilot_dataset import add_time_aware_features

SILVER = ROOT / "data" / "silver"

CONTEXT_COLUMNS = {
    "sector": "Setor do canal (catalogo)",
    "channel_type": "brand ou creator (catalogo)",
    "market": "Mercado do canal",
    "duration_seconds": "Duracao do Short",
    "duration_bucket": "Faixa de duracao: 01_15, 16_30, 31_60, 61_180",
    "publish_hour_utc": "Hora de publicacao (UTC)",
    "publish_weekday": "Dia da semana (0=segunda)",
    "publish_hour_sin": "Hora ciclica (seno)",
    "publish_hour_cos": "Hora ciclica (cosseno)",
    "publish_weekday_sin": "Dia ciclico (seno)",
    "publish_weekday_cos": "Dia ciclico (cosseno)",
    "is_weekend": "Sabado ou domingo",
    "channel_prior_posts": "Shorts elegiveis do canal publicados antes deste",
    "channel_prior_median_likes": "Mediana de curtidas dos Shorts anteriores do canal (base do B0)",
    "baseline_log_likes": "log1p da mediana anterior: a previsao B0",
    "days_since_previous_post": "Dias desde o Short anterior do canal",
    "channel_prior_mean_likes": "Media de curtidas anteriores",
    "channel_prior_like_std": "Desvio de curtidas anteriores",
    "channel_recent_5_median_likes": "Mediana dos 5 Shorts anteriores",
    "channel_recent_10_median_likes": "Mediana dos 10 Shorts anteriores",
    "channel_recent_5_like_trend": "Tendencia recente em log",
    "channel_recent_5_like_volatility": "Volatilidade recente em log",
    "channel_prior_median_views": "Mediana de views anteriores",
    "channel_prior_median_comments": "Mediana de comentarios anteriores",
    "channel_prior_like_view_rate": "Mediana de (likes+1)/(views+1) anteriores",
}
TEXT_COLUMNS = {
    "title_length": "Caracteres do titulo",
    "description_length": "Caracteres da descricao",
    "tag_count": "Quantidade de tags",
    "title_word_count": "Palavras do titulo",
    "description_word_count": "Palavras da descricao",
    "has_question": "Titulo/descricao tem ?",
    "exclamation_count": "Quantidade de !",
    "emoji_count": "Quantidade de emojis",
    "hashtag_count": "Quantidade de hashtags",
    "mention_count": "Quantidade de @mencoes",
    "url_count": "Quantidade de links",
    "digit_count": "Quantidade de digitos",
    "uppercase_ratio": "Proporcao de maiusculas",
    "has_cta": "Chamada para acao (curta, comente, salve...)",
    "title_tag_overlap": "Proporcao de palavras do titulo presentes nas tags",
}

# --- language detection for transcripts (no model download needed) ---------
STOPWORDS = {
    "pt": set("de que o a e do da em um uma para com não nao os no na por mais as dos como mas ao ele das à seu sua ou quando muito nos já ja eu também tambem só so pelo pela até ate isso ela entre depois sem mesmo aos seus quem nas me esse eles você voce essa num nem suas meu às minha numa pelos elas qual nós nos lhe deles essas esses pelas este dele tu te vocês voces vos lhes meus minhas teu tua teus tuas nosso nossa nossos nossas dela delas esta estes estas aquele aquela aqueles aquelas isto aquilo estou está esta estamos estão estao tá ta tô to aqui vai pra gente".split()),
    "es": set("de la que el en y a los del se las por un para con no una su al lo como más mas pero sus le ya o este sí si porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros".split()),
    "en": set("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were".split()),
    "fr": set("le la les de des du un une et est en que qui dans pour pas sur au aux avec ce cette ces il elle nous vous ils elles se sa son ses leur leurs plus mais ou donc ni car je tu on y être etre avoir fait c'est d'un d'une aujourd'hui notre nos votre vos".split()),
}
CTA_SPOKEN = re.compile(r"\b(se inscrev\w*|inscreva|curt[ae]|deix[ae] (o )?(seu )?like|coment[ae]\w*|compartilh\w*|link na bio|segue|siga|ativa o sininho|salv[ae]|subscribe|like (the|this) video|leave (a|your) comment|comment below|link in (the )?bio|follow (us|me)|share (this|it))\b", re.I)


def detect_language(text: str) -> tuple[str, float]:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    if len(words) < 5:
        return "unknown", 0.0
    scores = {lang: sum(w in sw for w in words) / len(words) for lang, sw in STOPWORDS.items()}
    lang = max(scores, key=scores.get)
    return (lang, round(scores[lang], 4)) if scores[lang] >= 0.08 else ("unknown", round(scores[lang], 4))


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() or "À" <= c <= "ɏ" for c in letters) / len(letters) if letters else 1.0


def load_eligible(silver_run_id: str) -> tuple[pd.DataFrame, Path]:
    path = SILVER / "eligible_shorts" / silver_run_id / "eligible_shorts.csv"
    data = pd.read_csv(path, dtype={"video_id": "string"})
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    for column in ("like_count", "view_count", "comment_count", "duration_seconds"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["tags"] = data["tags"].fillna("[]")
    return data, path


def build_context_and_text(data: pd.DataFrame, eligible_path: Path, silver_run_id: str, groups: list[str]) -> list[dict]:
    # Reuse the exact, already-reviewed logic of the pilot Gold so B0 and the
    # history features stay identical in definition.
    enriched = add_time_aware_features(data)
    out = []
    if "context" in groups:
        frame = enriched[["video_id", *CONTEXT_COLUMNS]].copy()
        out.append(write_feature_set(
            name="context_v1", group="A. Metadados e contexto", extractor_name="prepare_pilot_dataset.add_time_aware_features",
            extractor_version="1.0.0", frame=frame, sources=[eligible_path], params={"silver_run_id": silver_run_id},
            description="Contexto do canal e do post. Historico calculado so com Shorts estritamente anteriores do mesmo canal.",
            columns=CONTEXT_COLUMNS))
    if "text" in groups:
        frame = enriched[["video_id", *TEXT_COLUMNS]].copy()
        out.append(write_feature_set(
            name="text_v1", group="B. Texto", extractor_name="prepare_pilot_dataset.add_time_aware_features",
            extractor_version="1.0.0", frame=frame, sources=[eligible_path], params={"silver_run_id": silver_run_id},
            description="Atributos do titulo, descricao e tags publicados. Embedding textual fica para text_v2.",
            columns=TEXT_COLUMNS))
    return out


def latest_valid_run(base: Path) -> Path:
    runs = [p for p in sorted(base.iterdir()) if p.is_dir() and not (p / "SUPERSEDED.txt").exists()]
    if not runs:
        raise FileNotFoundError(base)
    return runs[-1]


def build_thumbnail(data: pd.DataFrame) -> dict:
    visual_path = latest_valid_run(SILVER / "features" / "thumbnail_v1") / "thumbnail_features.csv"
    ocr_path = latest_valid_run(SILVER / "features" / "thumbnail_ocr_v1") / "thumbnail_ocr_features.csv"
    visual = pd.read_csv(visual_path, dtype={"video_id": "string"})
    ocr = pd.read_csv(ocr_path, dtype={"video_id": "string"})
    visual_cols = ["is_vertical_pillarbox", "letterbox_fraction", "is_low_resolution", "brightness_mean", "brightness_std",
                   "saturation_mean", "colorfulness", "dark_pixel_pct", "light_pixel_pct", "edge_density", "dominant_hue"]
    ocr_cols = ["has_text", "ocr_word_count", "ocr_char_count", "ocr_text_area_pct", "ocr_uppercase_ratio",
                "ocr_has_digits", "ocr_has_question", "ocr_has_exclamation", "ocr_mean_confidence"]
    frame = data[["video_id"]].merge(visual[["video_id", *visual_cols]], on="video_id", how="left")
    frame = frame.merge(ocr[["video_id", *ocr_cols]], on="video_id", how="left")
    frame = frame.rename(columns={c: f"thumb_{c}" for c in visual_cols + ocr_cols})
    frame.insert(1, "has_thumbnail", frame["thumb_brightness_mean"].notna().astype(int))
    docs = {"has_thumbnail": "1 se a thumbnail foi baixada e validada"}
    docs.update({f"thumb_{c}": "ver docs/07, secao thumbnail_v1" for c in visual_cols})
    docs.update({f"thumb_{c}": "ver docs/07, secao thumbnail_ocr_v1 (EasyOCR)" for c in ocr_cols})
    return write_feature_set(
        name="thumbnail_v1", group="C. Thumbnail", extractor_name="extract_thumbnail_features + extract_thumbnail_ocr",
        extractor_version="1.1.0", frame=frame, sources=[visual_path, ocr_path],
        params={"pillarbox_crop": True, "ocr_engine": "easyocr-latin_g2", "ocr_min_confidence": 0.40},
        description="Atributos visuais e texto em tela da thumbnail, calculados no painel de conteudo (sem as laterais desfocadas dos Shorts).",
        coverage_column="has_thumbnail", columns=docs)


def build_transcript(data: pd.DataFrame, silver_run_id: str) -> dict:
    manifest_path = SILVER / "transcript_manifest" / silver_run_id / "transcript_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"video_id": "string"})
    rows, sources = [], [manifest_path]
    valid = manifest.loc[manifest["has_transcript"].astype(str).str.lower().eq("true")]
    for record in valid.itertuples(index=False):
        files = ast.literal_eval(record.subtitle_files) if isinstance(record.subtitle_files, str) else []
        if not files:
            continue
        path = ROOT / files[0]["local_path"]
        snippets = json.loads(path.read_text(encoding="utf-8")).get("snippets", [])
        text = " ".join(str(s.get("text", "")) for s in snippets).strip()
        words = re.findall(r"\w+", text)
        end = max((float(s.get("start", 0)) + float(s.get("duration", 0)) for s in snippets), default=0.0)
        lang, lang_score = detect_language(text)
        lat = latin_ratio(text)
        rows.append({
            "video_id": record.video_id,
            "transcript_provider": getattr(record, "provider", None) or "youtube-transcript-api",
            "transcript_language": lang if lat >= 0.7 else "non_latin",
            "transcript_language_score": lang_score,
            "transcript_word_count": len(words),
            "transcript_snippet_count": len(snippets),
            "transcript_speech_seconds": round(end, 2),
            "transcript_words_per_second": round(len(words) / end, 3) if end > 0 else np.nan,
            "transcript_hook_words_5s": sum(len(re.findall(r"\w+", str(s.get("text", "")))) for s in snippets if float(s.get("start", 0)) < 5),
            "transcript_question_count": text.count("?"),
            "transcript_has_spoken_cta": int(bool(CTA_SPOKEN.search(text))),
        })
    feats = pd.DataFrame(rows)
    frame = data[["video_id"]].merge(feats, on="video_id", how="left") if len(feats) else data[["video_id"]].copy()
    frame.insert(1, "has_transcript", frame.get("transcript_word_count", pd.Series(dtype=float)).notna().astype(int))
    frame.insert(2, "transcript_is_portuguese", frame.get("transcript_language", pd.Series(dtype=str)).eq("pt").astype(int))
    # Channels in the pilot speak Portuguese; an English/French track is a provider auto-translation,
    # so counts and timing stay usable but language-specific features are approximate.
    frame.insert(3, "transcript_is_auto_translation", (frame["has_transcript"].eq(1) & frame["transcript_language"].isin(["en", "es", "fr"])).astype(int))
    docs = {
        "has_transcript": "1 se ha legenda valida (transcript_status=downloaded e arquivo integro)",
        "transcript_is_portuguese": "1 se o idioma detectado na fala e portugues; outras linguas = dublagem/legenda automatica de outro mercado",
        "transcript_is_auto_translation": "1 quando a legenda veio traduzida (en/es/fr) pelo provider; o Apify devolveu traducao automatica em 121 de 323 casos",
        "transcript_language": "Idioma detectado por stopwords (pt, es, en, fr, unknown, non_latin)",
        "transcript_word_count": "Palavras faladas",
        "transcript_speech_seconds": "Fim do ultimo trecho com fala (s)",
        "transcript_words_per_second": "Densidade de fala",
        "transcript_hook_words_5s": "Palavras nos 5 primeiros segundos (gancho)",
        "transcript_question_count": "Perguntas na fala",
        "transcript_has_spoken_cta": "CTA falado (se inscreve, comenta, link na bio... e equivalentes em ingles para as traducoes)",
    }
    return write_feature_set(
        name="transcript_v1", group="D. Transcricao", extractor_name="build_feature_store.build_transcript",
        extractor_version="1.1.0", frame=frame, sources=sources,
        params={"silver_run_id": silver_run_id, "language_method": "stopwords_pt_es_en_fr", "min_stopword_share": 0.08},
        description="Atributos da fala a partir das legendas (sem baixar video). Usar apenas onde has_transcript=1; embedding da fala fica para transcript_v2.",
        coverage_column="has_transcript", columns=docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--silver-run-id", required=True)
    parser.add_argument("--groups", nargs="+", default=["context", "text", "thumbnail", "transcript"],
                        choices=["context", "text", "thumbnail", "transcript"])
    args = parser.parse_args()
    data, eligible_path = load_eligible(args.silver_run_id)
    results = build_context_and_text(data, eligible_path, args.silver_run_id, args.groups)
    if "thumbnail" in args.groups:
        results.append(build_thumbnail(data))
    if "transcript" in args.groups:
        results.append(build_transcript(data, args.silver_run_id))
    for meta in results:
        print(f"{meta['feature_set_id']:32s} rows={meta['rows']:5d} cols={len(meta['feature_columns']):3d} coverage={meta['coverage_rate']:.1%}")


if __name__ == "__main__":
    main()
