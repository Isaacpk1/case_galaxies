"""Coleta metadados e metricas de Shorts pela YouTube Data API v3 (so biblioteca padrao).

Por que existe: o Windows do Inteli bloqueia as DLLs de pandas/numpy, entao o
collect_data_discovery.py nao abre la. Este coletor grava o mesmo layout Bronze
(channels.json, videos.json, errors.json) e mais:
  - sem o teto de 300 uploads: pagina a playlist de uploads ate sair da janela
    de 18 meses (--max-uploads e so uma trava de seguranca);
  - aceita channel_id ou handle (@nome) no CSV de candidatos;
  - grava apos cada canal e, com --resume-run-id, pula canais ja coletados;
  - escreve channel_profile.csv / eligible_shorts.csv (Silver data_discovery)
    com a mesma regra de elegibilidade do coletor original.

Custo de cota: ~1 unidade por 50 uploads (playlistItems) + 1 por 50 videos
(videos.list) + 1 por canal. 60 canais com ~600 uploads ~ 1.500 unidades.

Uso (PowerShell, na raiz do repo):
  python scripts/coleta/collect_youtube_stdlib.py --candidates data/candidates/channels_v2.csv
  python scripts/coleta/collect_youtube_stdlib.py --candidates data/candidates/channels_v2.csv --resume-run-id <RUN>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.googleapis.com/youtube/v3"
MAX_SHORT_DURATION_SECONDS = 180
MIN_POST_AGE_DAYS = 30
HISTORY_DAYS = 548  # ~18 meses
MIN_SHORTS_PER_CHANNEL = 30
PREFERRED_SHORTS_PER_CHANNEL = 50


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = PROJECT_ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


API_KEY = load_env().get("YOUTUBE_API_KEY")
QUOTA = {"units": 0}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def youtube_get(resource: str, **params: object) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in {**params, "key": API_KEY}.items() if v is not None})
    url = f"{BASE_URL}/{resource}?{query}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as response:
                QUOTA["units"] += 100 if resource == "search" else 1
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error", {}).get("message", body[:300])
            except json.JSONDecodeError:
                message = body[:300]
            if error.code == 403 and "quota" in message.lower():
                raise SystemExit(f"Cota diaria da YouTube API esgotada: {message}")
            if error.code >= 500 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"YouTube API {error.code}: {message}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 3:
                raise RuntimeError("Erro de rede apos 4 tentativas.") from None
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def duration_seconds(value: str) -> int:
    match = re.fullmatch(r"P(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_channel(candidate: dict[str, str]) -> dict:
    channel_id = (candidate.get("channel_id") or "").strip()
    handle = (candidate.get("handle") or "").strip()
    params = {"part": "snippet,statistics,contentDetails"}
    if channel_id:
        params["id"] = channel_id
    elif handle:
        params["forHandle"] = handle if handle.startswith("@") else f"@{handle}"
    else:
        raise RuntimeError("linha sem channel_id nem handle")
    items = youtube_get("channels", **params).get("items", [])
    if not items:
        raise RuntimeError(f"canal nao encontrado ({channel_id or handle})")
    return items[0]


def collect_channel(candidate: dict[str, str], max_uploads: int, history_start: datetime) -> tuple[dict, list[dict]]:
    channel = resolve_channel(candidate)
    channel_id = channel["id"]
    uploads_playlist_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids: list[str] = []
    page_token: str | None = None
    while len(video_ids) < max_uploads:
        payload = youtube_get("playlistItems", part="contentDetails", playlistId=uploads_playlist_id,
                              maxResults=50, pageToken=page_token)
        items = payload.get("items", [])
        video_ids.extend(item["contentDetails"]["videoId"] for item in items)
        page_token = payload.get("nextPageToken")
        published = [parse_time(item["contentDetails"].get("videoPublishedAt")) for item in items]
        published = [p for p in published if p]
        # A playlist de uploads vem do mais novo para o mais antigo; parar quando a pagina inteira saiu da janela.
        if not page_token or (published and max(published) < history_start):
            break
    video_items: list[dict] = []
    for start in range(0, len(video_ids), 50):
        payload = youtube_get("videos", part="snippet,contentDetails,statistics,status",
                              id=",".join(video_ids[start:start + 50]))
        video_items.extend(payload.get("items", []))

    collected_at = datetime.now(timezone.utc).isoformat()
    snippet = channel.get("snippet", {})
    channel_record = {
        "channel_id": channel_id,
        "channel_name": candidate.get("channel_name") or snippet.get("title", ""),
        "sector": candidate.get("sector", ""),
        "market": candidate.get("market", "BR"),
        "channel_type": candidate.get("channel_type", "unknown"),
        "notes": candidate.get("notes", ""),
        "handle": snippet.get("customUrl", candidate.get("handle", "")),
        "subscriber_count_at_collection": channel.get("statistics", {}).get("subscriberCount"),
        "video_count_at_collection": channel.get("statistics", {}).get("videoCount"),
        "country": snippet.get("country"),
        "uploads_playlist_id": uploads_playlist_id,
        "uploads_scanned": len(video_ids),
        "collected_at": collected_at,
    }
    records: list[dict] = []
    for video in video_items:
        vs = video.get("snippet", {})
        details = video.get("contentDetails", {})
        stats = video.get("statistics", {})
        thumbs = vs.get("thumbnails", {})
        thumb = thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
        records.append({
            "video_id": video["id"],
            "channel_id": channel_id,
            "channel_name": channel_record["channel_name"],
            "sector": channel_record["sector"],
            "market": channel_record["market"],
            "channel_type": channel_record["channel_type"],
            "candidate_notes": channel_record["notes"],
            "published_at": vs.get("publishedAt"),
            "title": vs.get("title", ""),
            "description": vs.get("description", ""),
            "tags": json.dumps(vs.get("tags", []), ensure_ascii=False),
            "duration_seconds": duration_seconds(details.get("duration", "")),
            "privacy_status": video.get("status", {}).get("privacyStatus"),
            "view_count": stats.get("viewCount"),
            "like_count": stats.get("likeCount"),
            "comment_count": stats.get("commentCount"),
            "thumbnail_url": thumb.get("url"),
            "collected_at": collected_at,
        })
    return channel_record, records


def is_eligible(record: dict, history_start: datetime, cutoff: datetime) -> bool:
    published = parse_time(record.get("published_at"))
    return (record.get("privacy_status") == "public"
            and published is not None and history_start <= published <= cutoff
            and 1 <= int(record.get("duration_seconds") or 0) <= MAX_SHORT_DURATION_SECONDS
            and record.get("like_count") not in (None, ""))


def write_profile(channels: list[dict], videos: list[dict], silver_dir: Path,
                  history_start: datetime, cutoff: datetime) -> list[dict]:
    eligible = [v for v in videos if is_eligible(v, history_start, cutoff)]
    if eligible:
        with (silver_dir / "eligible_shorts.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(eligible[0].keys()))
            writer.writeheader()
            writer.writerows(eligible)
    by_channel: dict[str, list[dict]] = {}
    for v in eligible:
        by_channel.setdefault(v["channel_id"], []).append(v)
    profiles = []
    for c in channels:
        rows = by_channel.get(c["channel_id"], [])
        likes = [int(r["like_count"]) for r in rows]
        dates = sorted(r["published_at"] for r in rows)
        profiles.append({**c, "eligible_shorts": len(rows),
                         "median_likes": statistics.median(likes) if likes else "",
                         "mean_likes": round(statistics.mean(likes), 1) if likes else "",
                         "first_short": dates[0] if dates else "", "last_short": dates[-1] if dates else "",
                         "meets_minimum_history": len(rows) >= MIN_SHORTS_PER_CHANNEL,
                         "meets_preferred_history": len(rows) >= PREFERRED_SHORTS_PER_CHANNEL})
    if profiles:
        with (silver_dir / "channel_profile.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(profiles[0].keys()))
            writer.writeheader()
            writer.writerows(profiles)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta YouTube Shorts (so biblioteca padrao).")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "candidates" / "channels_v2.csv")
    parser.add_argument("--sectors", nargs="+", default=["moda", "automotivo", "beleza"])
    parser.add_argument("--max-uploads", type=int, default=3000, help="Trava de seguranca por canal")
    parser.add_argument("--resume-run-id", help="Continua um run anterior, pulando canais ja coletados")
    parser.add_argument("--only", nargs="*", help="Coletar so estes channel_id/handles")
    args = parser.parse_args()
    if not API_KEY:
        raise SystemExit("Defina YOUTUBE_API_KEY no .env antes de rodar.")

    candidates_path = args.candidates if args.candidates.is_absolute() else PROJECT_ROOT / args.candidates
    with candidates_path.open(encoding="utf-8-sig", newline="") as stream:
        candidates = [r for r in csv.DictReader(stream) if r.get("sector") in args.sectors]
    if args.only:
        wanted = set(args.only)
        candidates = [c for c in candidates if c.get("channel_id") in wanted or c.get("handle") in wanted]

    run_id = args.resume_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "youtube" / run_id
    silver_dir = PROJECT_ROOT / "data" / "silver" / "data_discovery" / run_id
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> list:
        path = bronze_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    channel_records, video_records, errors = load("channels.json"), load("videos.json"), []
    done = {c["channel_id"] for c in channel_records} | {c.get("handle", "") for c in channel_records}
    now = datetime.now(timezone.utc)
    history_start, cutoff = now - timedelta(days=HISTORY_DAYS), now - timedelta(days=MIN_POST_AGE_DAYS)
    write_json(bronze_dir / "run_config.json", {"run_id": run_id, "candidates": str(candidates_path.name),
               "history_start": history_start.isoformat(), "cutoff": cutoff.isoformat(),
               "max_uploads": args.max_uploads, "collector": "collect_youtube_stdlib.py"})

    for i, candidate in enumerate(candidates, 1):
        key = candidate.get("channel_id") or candidate.get("handle")
        if key in done or ("@" + key.lstrip("@")) in done:
            print(f"[{i}/{len(candidates)}] {candidate.get('channel_name')}: ja coletado, pulando")
            continue
        print(f"[{i}/{len(candidates)}] {candidate.get('channel_name') or key}...", end=" ", flush=True)
        try:
            channel_record, records = collect_channel(candidate, args.max_uploads, history_start)
            channel_record["sector"] = channel_record["sector"] or candidate.get("sector", "")
            channel_records.append(channel_record)
            video_records.extend(records)
            n_ok = sum(is_eligible(r, history_start, cutoff) for r in records)
            print(f"{len(records)} uploads, {n_ok} Shorts elegiveis (cota usada: {QUOTA['units']})")
        except RuntimeError as error:
            errors.append({"key": key, "channel_name": candidate.get("channel_name"), "error": str(error)})
            print(f"ERRO: {error}")
        write_json(bronze_dir / "channels.json", channel_records)
        write_json(bronze_dir / "videos.json", video_records)
        write_json(bronze_dir / "errors.json", errors)

    profiles = write_profile(channel_records, video_records, silver_dir, history_start, cutoff)
    ok = [p for p in profiles if p["meets_minimum_history"]]
    print(f"\nRun {run_id}: {len(channel_records)} canais, {len(video_records)} uploads, erros {len(errors)}.")
    print(f"Canais com >= {MIN_SHORTS_PER_CHANNEL} Shorts: {len(ok)}; Shorts elegiveis nesses canais: "
          f"{sum(p['eligible_shorts'] for p in ok)}. Cota usada: ~{QUOTA['units']} unidades.")


if __name__ == "__main__":
    sys.exit(main())
