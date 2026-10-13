"""Attempt caption-only collection and record every outcome.

This is an exploratory collector, not a video downloader. Its default provider
retrieves generated captions through youtube-transcript-api; yt-dlp can be used
as an explicit fallback with --skip-download. Both paths store only transcript
files and make rate limits, access blocks and unavailable captions visible in a
manifest.

Anti-blocking design (no accounts, no proxies):
- paced requests: a jittered pause between every video, not only after errors;
- circuit breaker: the run stops after N consecutive blocks instead of
  hammering YouTube (which extends an IP ban);
- resumable: videos already downloaded (or known to have no captions) in any
  previous run are skipped, and videos not attempted are saved as pending;
- one shared HTTP client for the whole run.

Providers:
- transcript-api (default) and yt-dlp fetch from YouTube using YOUR IP;
- apify sends video URLs to a third-party Apify actor, which fetches the captions
  from its own infrastructure. Requires APIFY_TOKEN in .env and costs credits.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import random
import subprocess
import sys
import time
import csv
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

# Third-party imports are lazy: the Inteli Windows Application Control policy blocks
# native DLLs (pandas, numpy, charset_normalizer...). With --provider apify this
# script runs on the standard library alone.


def dotenv_values(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE per line) so python-dotenv is not required."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _transcript_api():
    from youtube_transcript_api import YouTubeTranscriptApi
    return YouTubeTranscriptApi()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_ROOT = PROJECT_ROOT / "data" / "bronze" / "transcripts" / "youtube_caption_pilot"
USAGE_NOTE = "Exploratory collection. Do not treat as a licensed-media source without separate authorization review."
APIFY_USAGE_NOTE = (
    "Exploratory collection via a third-party Apify actor, which retrieves captions from its own "
    "infrastructure. Do not treat as a licensed-media source without separate authorization review."
)
APIFY_DEFAULT_ACTOR = "karamelo/youtube-transcripts"  # batch, fast; may return an auto-dubbed track
# Per-video fallback with explicit language, used only when the main actor returns a non-Latin (dubbed) track.
APIFY_FALLBACK_ACTOR = "pintostudio/youtube-transcript-scraper"
APIFY_DEFAULT_PRICE_PER_VIDEO_USD = 0.005  # "from $5.00 / 1,000 transcripts" (check the actor page)

# youtube-transcript-api exception classes (checked through the MRO, so subclasses count).
BLOCKED_ERRORS = {"RequestBlocked", "IpBlocked", "PoTokenRequired"}
NOT_AVAILABLE_ERRORS = {
    "NoTranscriptFound", "TranscriptsDisabled", "VideoUnavailable",
    "VideoUnplayable", "AgeRestricted", "InvalidVideoId",
}
# yt-dlp output markers for an access challenge. Avoid bare "bot" (matches "both", "robots").
YTDLP_BLOCK_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "http error 403",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    """Persist an append-only operational event for monitoring and diagnosis."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_record(video_id: str, run_id: str, provider: str, languages: list[str]) -> dict:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "collection_mode": "captions_only_no_video_download",
        "provider": provider,
        "caption_source_requested": "automatic_captions",
        "languages_requested": languages,
        "collected_at": now(),
        "subtitle_files": [],
        "usage_note": USAGE_NOTE,
    }


def status_from(output: str, subtitle_files: list[Path]) -> tuple[str, str | None]:
    text = output.lower()
    if subtitle_files:
        return "downloaded", None
    if "http error 429" in text or "too many requests" in text:
        return "rate_limited", "youtube_http_429"
    if "there are no subtitles" in text or "has no subtitles" in text:
        return "not_available", "no_subtitles"
    if any(marker in text for marker in YTDLP_BLOCK_MARKERS):
        return "blocked", "youtube_access_challenge"
    return "failed", "yt_dlp_error"


def transcript_api_status(error: Exception) -> tuple[str, str]:
    name = type(error).__name__
    lineage = {cls.__name__ for cls in type(error).__mro__}
    if lineage & BLOCKED_ERRORS:
        return "blocked", name
    if "429" in str(error) or "Too Many Requests" in str(error):
        return "rate_limited", name
    if lineage & NOT_AVAILABLE_ERRORS:
        return "not_available", name
    return "failed", name


def collect_with_transcript_api(api: YouTubeTranscriptApi, video_id: str, files_dir: Path, languages: list[str], run_id: str) -> dict:
    record = base_record(video_id, run_id, "youtube_transcript_api", languages)
    try:
        transcript = api.list(video_id).find_generated_transcript(languages).fetch()
    except Exception as error:  # Provider exception classes vary by release.
        status, error_code = transcript_api_status(error)
        return {**record, "transcript_status": status, "error_code": error_code, "diagnostic_tail": str(error)[-1500:]}

    destination = files_dir / f"{video_id}.{transcript.language_code}.generated.json"
    snippets = [{"text": item.text, "start": item.start, "duration": item.duration} for item in transcript]
    write_json(destination, {
        "video_id": video_id,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "snippets": snippets,
    })
    return {
        **record,
        "transcript_status": "downloaded",
        "error_code": None,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "snippet_count": len(snippets),
        "subtitle_files": [{
            "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }],
    }


class _LogCollector:
    """Collect yt-dlp log lines so they can be stored as diagnostic_tail."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def debug(self, message: str) -> None:
        if not message.startswith("[debug] "):
            self.lines.append(message)

    def info(self, message: str) -> None:
        self.lines.append(message)

    def warning(self, message: str) -> None:
        self.lines.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.lines.append(f"ERROR: {message}")


def pick_caption_track(info: dict, languages: list[str]) -> str | None:
    """Pick exactly ONE automatic caption track: the video's original ASR track.

    Videos with auto-dubbing expose many "<lang>-orig" tracks (one per dubbed audio)
    plus ~100 auto-translations. Requesting several of them triggers HTTP 429, so
    we request only the original-language track, falling back to preferred languages.
    """
    tracks = info.get("automatic_captions") or {}
    video_language = info.get("language")
    candidates: list[str] = []
    if video_language:
        candidates += [f"{video_language}-orig", video_language]
    for language in languages:
        candidates += [f"{language}-orig"]
    base_preferred = {language.split("-")[0] for language in languages}
    candidates += sorted(k for k in tracks if k.endswith("-orig") and k.split("-")[0] in base_preferred)
    return next((track for track in candidates if track in tracks), None)


def collect_with_ytdlp(video_id: str, files_dir: Path, languages: list[str], run_id: str) -> dict:
    """One metadata request + one subtitle request per video, via the yt-dlp Python API."""
    import yt_dlp
    from yt_dlp.networking.exceptions import HTTPError

    record = base_record(video_id, run_id, "yt_dlp", languages)
    logger = _LogCollector()
    options = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "sleep_interval_requests": 2,
        "logger": logger,
        # The "web" client is the one that gets a subtitles PO Token from the bgutil
        # provider (http://127.0.0.1:4416). Without it the caption endpoint answers 429.
        "extractor_args": {"youtube": {"player_client": ["web"]}},
        # web client exposes no downloadable video formats (SABR); we only need captions.
        "ignore_no_formats_error": True,
    }

    def failure(status: str, error_code: str, extra: dict | None = None) -> dict:
        return {**record, "transcript_status": status, "error_code": error_code, **(extra or {}),
                "diagnostic_tail": "\n".join(logger.lines)[-1500:]}

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(record["youtube_url"], download=False)
            track = pick_caption_track(info, languages)
            meta = {"video_language": info.get("language"), "caption_track": track,
                    "available_orig_tracks": sorted(k for k in (info.get("automatic_captions") or {}) if k.endswith("-orig"))}
            if track is None:
                return failure("not_available", "no_subtitles", meta)
            formats = info["automatic_captions"][track]
            vtt = next((f for f in formats if f.get("ext") == "vtt"), formats[0])
            time.sleep(2)
            content = ydl.urlopen(vtt["url"]).read()
    except HTTPError as error:
        status = "rate_limited" if error.status == 429 else "blocked" if error.status == 403 else "failed"
        logger.error(str(error))
        return failure(status, f"youtube_http_{error.status}", locals().get("meta"))
    except yt_dlp.utils.DownloadError as error:
        status, error_code = status_from(str(error) + "\n" + "\n".join(logger.lines), [])
        return failure(status, error_code)

    destination = files_dir / f"{video_id}.{track}.{vtt.get('ext', 'vtt')}"
    destination.write_bytes(content)
    return {
        **record,
        **meta,
        "transcript_status": "downloaded",
        "error_code": None,
        "language_code": track.removesuffix("-orig"),
        "subtitle_files": [{
            "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }],
        "diagnostic_tail": "\n".join(logger.lines)[-1500:],
    }


def collect_with_rate_limit_retry(
    api: YouTubeTranscriptApi,
    video_id: str,
    files_dir: Path,
    languages: list[str],
    run_id: str,
    provider: str,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
) -> dict:
    """Retry only temporary rate limiting; never retry access blocks."""
    delays: list[float] = []
    for attempt in range(max_retries + 1):
        try:
            record = (
                collect_with_transcript_api(api, video_id, files_dir, languages, run_id)
                if provider == "transcript-api"
                else collect_with_ytdlp(video_id, files_dir, languages, run_id)
            )
        except subprocess.TimeoutExpired:
            record = {**base_record(video_id, run_id, provider, languages), "transcript_status": "failed", "error_code": "timeout"}
        if record["transcript_status"] != "rate_limited" or attempt == max_retries:
            return {**record, "attempts": attempt + 1, "retry_delays_seconds": delays}
        delay = min(backoff_max_seconds, backoff_base_seconds * (2**attempt)) + random.uniform(0, backoff_base_seconds)
        delays.append(round(delay, 3))
        print(f"  rate_limited; retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_apify_captions(captions: object) -> list[dict]:
    """Convert the actor's `captions` field (XML string, list of strings or list of dicts) to snippets."""
    if captions is None:
        return []
    if isinstance(captions, list) and captions and all(isinstance(c, str) for c in captions) \
            and captions[0].lstrip().startswith("<"):
        captions = "".join(captions)
    if isinstance(captions, str):
        text = captions.strip()
        matches = re.findall(r'<text start="([^"]+)"(?: dur="([^"]*)")?[^>]*>(.*?)</text>', text, flags=re.S)
        if matches:
            return [
                {"text": html.unescape(html.unescape(re.sub(r"<[^>]+>", "", body))).strip(),  # YouTube XML is often double-escaped
                 "start": _float_or_none(start), "duration": _float_or_none(duration)}
                for start, duration, body in matches
                if body.strip()
            ]
        return [{"text": text, "start": None, "duration": None}] if text else []
    snippets: list[dict] = []
    if isinstance(captions, list):
        for item in captions:
            if isinstance(item, str) and item.strip():
                snippets.append({"text": item.strip(), "start": None, "duration": None})
            elif isinstance(item, dict) and str(item.get("text", "")).strip():
                snippets.append({
                    "text": str(item["text"]).strip(),
                    "start": _float_or_none(item.get("start", item.get("offset", item.get("startTime")))),
                    "duration": _float_or_none(item.get("dur", item.get("duration"))),
                })
    return snippets


def latin_ratio(text: str) -> float:
    """Share of alphabetic characters in Latin script; pt/es/en ~1.0, Arabic/Hindi/Japanese ~0."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() or "\u00c0" <= c <= "\u024f" for c in letters) / len(letters)


# Stopword shares are enough to tell pt from a provider's auto-translation (en/es/fr):
# the karamelo actor returned English translations of Portuguese speech (D-012).
_STOPWORDS = {
    "pt": set("de que o a e do da em um uma para com não nao os no na por mais as dos como mas ao ele das seu sua ou quando muito nos já ja eu também tambem só so pelo pela até ate isso ela entre depois sem mesmo você voce essa esse está esta estou tá ta aqui vai pra gente".split()),
    "es": set("de la que el en y los del se las por un para con no una su al lo como más pero sus le ya este porque esta entre cuando muy sin sobre también hasta hay donde desde todo nos".split()),
    "en": set("the be to of and a in that have it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like just know your some them see than then now only its over think also back after how our well way because".split()),
    "fr": set("le la les de des du un une et est en que qui dans pour pas sur au aux avec ce cette ces il elle nous vous ils se sa son ses leur plus mais ou donc je tu on être fait".split()),
}


def detect_language(text: str) -> str:
    """'pt', 'es', 'en', 'fr' or 'unknown' (music-only or too short)."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    if len(words) < 5:
        return "unknown"
    scores = {lang: sum(w in sw for w in words) / len(words) for lang, sw in _STOPWORDS.items()}
    lang = max(scores, key=scores.get)
    return lang if scores[lang] >= 0.08 else "unknown"


def is_wrong_language(text: str, languages: list[str]) -> bool:
    """Non-Latin dubbing, or a Latin track in a language other than the first preferred one."""
    if latin_ratio(text) < 0.7:
        return True
    detected = detect_language(text)
    return detected not in ("unknown", languages[0].split("-")[0])


def _apify_language(item: dict) -> str | None:
    for key in ("languageCode", "language_code", "lang", "language"):
        if item.get(key):
            return str(item[key])
    return None


def _apify_video_id(item: dict) -> str | None:
    for key in ("videoId", "video_id", "id"):
        if item.get(key) and len(str(item[key])) == 11:
            return str(item[key])
    for key in ("url", "inputUrl", "videoUrl"):
        match = re.search(r"(?:v=|shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", str(item.get(key, "")))
        if match:
            return match.group(1)
    return None


APIFY_API = "https://api.apify.com/v2"
APIFY_TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}


def _apify_request(method: str, url: str, token: str, params: dict | None = None, json: dict | None = None) -> dict | list:
    """Plain REST call with urllib (no apify-client / requests: their native DLLs can be blocked on Windows)."""
    import json as _json

    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = _json.dumps(json).encode("utf-8") if json is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = _json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise RuntimeError("Apify rejected APIFY_TOKEN (401). Check the token in .env.") from error
        if error.code == 402:
            raise RuntimeError("Apify: not enough credits (402).") from error
        raise
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


def _run_apify_actor(token: str, actor_id: str, run_input: dict, max_items: int, max_wait_seconds: int = 900) -> tuple[dict, list]:
    actor_path = actor_id.replace("/", "~")
    run = _apify_request("POST", f"{APIFY_API}/acts/{actor_path}/runs", token,
                         params={"waitForFinish": 60, "maxItems": max_items}, json=run_input)
    waited = 60
    while run.get("status") not in APIFY_TERMINAL and waited < max_wait_seconds:
        print(f"  Apify run {run.get('id')} {run.get('status')}; waiting...", flush=True)
        run = _apify_request("GET", f"{APIFY_API}/actor-runs/{run['id']}", token, params={"waitForFinish": 60})
        waited += 60
    dataset_id = run.get("defaultDatasetId")
    items = _apify_request("GET", f"{APIFY_API}/datasets/{dataset_id}/items", token,
                           params={"clean": "true", "format": "json"}) if dataset_id else []
    return run, items


def _items_to_snippets(items: list) -> list[dict]:
    """pintostudio-style output: either one item per segment or one item holding a list."""
    if items and all(isinstance(i, dict) and "text" in i and ("start" in i or "dur" in i) for i in items):
        return normalize_apify_captions(items)
    for item in items:
        for key in ("data", "transcript", "captions", "segments"):
            if isinstance(item, dict) and item.get(key):
                return normalize_apify_captions(item[key])
    return []


def apify_language_fallback(
    token: str, actor_id: str, video_id: str, language: str, files_dir: Path, raw_dir: Path, base: dict,
) -> dict | None:
    """Retry one video with an actor that takes an explicit language. Returns a downloaded record or None."""
    run_input = {"videoUrl": f"https://www.youtube.com/watch?v={video_id}", "targetLanguage": language}
    run, items = _run_apify_actor(token, actor_id, run_input, max_items=5000, max_wait_seconds=600)
    write_json(raw_dir / f"apify_fallback_{video_id}_{run.get('id')}.json", {
        "actor_id": actor_id, "apify_run_id": run.get("id"), "apify_status": run.get("status"),
        "usage_total_usd": run.get("usageTotalUsd"), "run_input": run_input, "items": items,
    })
    snippets = _items_to_snippets(items)
    if not snippets or is_wrong_language(" ".join(x["text"] for x in snippets), [language]):
        return None
    destination = files_dir / f"{video_id}.apify.json"
    write_json(destination, {"video_id": video_id, "language": None, "language_code": language,
                             "is_generated": None, "snippets": snippets})
    return {
        **base, "transcript_status": "downloaded", "error_code": None,
        "apify_actor": actor_id, "apify_run_id": run.get("id"), "apify_fallback_from": base.get("apify_actor"),
        "language_code": language, "snippet_count": len(snippets),
        "has_timestamps": any(x["start"] is not None for x in snippets),
        "subtitle_files": [{
            "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": destination.stat().st_size, "sha256": sha256(destination),
        }],
    }


def collect_batch_with_apify(
    token: str, actor_id: str, output_format: str, video_ids: list[str],
    files_dir: Path, raw_dir: Path, languages: list[str], run_id: str,
    fallback_actor: str | None = None,
) -> list[dict]:
    """Run the Apify actor once for a batch of videos and return one record per video."""
    urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids]
    if actor_id.startswith("karamelo/"):
        run_input = {"urls": urls, "outputFormat": output_format, "maxRetries": 8}
    else:
        run_input = {"urls": urls, "preferredLanguages": list(dict.fromkeys(l.split("-")[0] for l in languages))}
    run, items = _run_apify_actor(token, actor_id, run_input, max_items=len(urls))
    run_status = run.get("status")
    write_json(raw_dir / f"apify_run_{run.get('id')}.json", {
        "actor_id": actor_id, "apify_run_id": run.get("id"), "apify_status": run_status,
        "usage_total_usd": run.get("usageTotalUsd"), "run_input": run_input, "items": items,
    })
    by_video = {}
    for item in items:
        video_id = _apify_video_id(item)
        if video_id:
            by_video.setdefault(video_id, item)

    records = []
    for video_id in video_ids:
        record = {**base_record(video_id, run_id, "apify", languages), "usage_note": APIFY_USAGE_NOTE,
                  "apify_actor": actor_id, "apify_run_id": run.get("id"), "apify_output_format": output_format}
        item = by_video.get(video_id)
        if item is None:
            records.append({**record, "transcript_status": "failed",
                            "error_code": f"apify_no_item_{str(run_status or 'unknown').lower()}"})
            continue
        if item.get("error"):
            detail = str(item.get("detail") or item.get("error"))
            status = "rate_limited" if "429" in detail else "blocked" if "403" in detail else "failed"
            records.append({**record, "transcript_status": status, "error_code": f"apify_{item['error']}",
                            "diagnostic_tail": detail[-1500:]})
            continue
        caption_field = next((item[k] for k in ("captions", "segments", "transcript", "subtitles", "text")
                              if item.get(k)), None)
        snippets = normalize_apify_captions(caption_field)
        if not snippets:
            reason = item.get("reason") or item.get("status") or "no_captions"
            records.append({**record, "transcript_status": "not_available", "error_code": "apify_no_transcript",
                            "diagnostic_tail": str(reason)[-1500:]})
            continue
        full_text = " ".join(snippet["text"] for snippet in snippets)
        latin = latin_ratio(full_text)
        if is_wrong_language(full_text, languages):  # dubbed (e.g. Arabic) or auto-translated (e.g. English) track
            detected = detect_language(full_text) if latin >= 0.7 else "non_latin"
            write_json(raw_dir / f"{video_id}.wrong_language.json", {"video_id": video_id, "latin_ratio": latin, "detected_language": detected, "snippets": snippets})
            if fallback_actor:
                print(f"  {video_id}: wrong-language track ({detected}); retrying with {fallback_actor}...", flush=True)
                try:
                    fixed = apify_language_fallback(token, fallback_actor, video_id, languages[0].split("-")[0],
                                                    files_dir, raw_dir, record)
                except Exception as error:  # fallback is best-effort
                    fixed = None
                    print(f"  fallback error: {type(error).__name__}: {str(error)[:200]}", flush=True)
                if fixed:
                    records.append(fixed)
                    continue
            records.append({**record, "transcript_status": "failed", "error_code": "apify_wrong_language",
                            "latin_ratio": round(latin, 3), "detected_language": detected, "diagnostic_tail": snippets[0]["text"][:300]})
            continue
        destination = files_dir / f"{video_id}.apify.json"
        write_json(destination, {
            "video_id": video_id,
            "language": item.get("language"),
            "language_code": _apify_language(item),
            "is_generated": item.get("isGenerated", item.get("is_generated", item.get("isAutoGenerated"))),
            "title": item.get("title"),
            "snippets": snippets,
        })
        records.append({
            **record,
            "transcript_status": "downloaded",
            "error_code": None,
            "language_code": _apify_language(item),
            "snippet_count": len(snippets),
            "has_timestamps": any(snippet["start"] is not None for snippet in snippets),
            "subtitle_files": [{
                "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }],
        })
    return records


def _file_is_latin(record: dict) -> bool:
    """False for JSON transcripts that hold an auto-dubbed non-Latin track (e.g. Arabic)."""
    path = PROJECT_ROOT / record["subtitle_files"][0]["local_path"]
    if path.suffix != ".json":
        return True
    try:
        snippets = json.loads(path.read_text(encoding="utf-8")).get("snippets", [])
    except (json.JSONDecodeError, OSError):
        return False
    text = " ".join(str(x.get("text", "")) for x in snippets[:50])
    # Also re-collect auto-translated tracks (D-012); the default preferred language is Portuguese.
    return latin_ratio(text) >= 0.7 and detect_language(text) in ("pt", "unknown")


def known_outcomes(include_not_available: bool) -> dict[str, str]:
    """Scan previous runs: video_id -> 'downloaded' or 'not_available' (final outcomes worth skipping)."""
    outcomes: dict[str, str] = {}
    for summary in sorted(TRANSCRIPT_ROOT.glob("*/run_summary.json")):
        try:
            records = json.loads(summary.read_text(encoding="utf-8")).get("records", [])
        except (json.JSONDecodeError, OSError):
            continue
        for record in records:
            status = record.get("transcript_status")
            files_exist = all((PROJECT_ROOT / f["local_path"]).exists() for f in record.get("subtitle_files", []))
            if status == "downloaded" and record.get("subtitle_files") and files_exist and _file_is_latin(record):
                outcomes[record["video_id"]] = "downloaded"
            elif status == "not_available" and include_not_available:
                outcomes.setdefault(record["video_id"], "not_available")
    return outcomes


def stratified_sample(rows: list[dict], total: int, min_per_channel: int, seed: int) -> list[dict]:
    """Proportional allocation by channel_id with a per-channel floor. Same seed -> same sample (resumable)."""
    rng = random.Random(seed)
    by_channel: dict[str, list[dict]] = {}
    for row in rows:
        by_channel.setdefault(row.get("channel_id") or "", []).append(row)
    picked: list[dict] = []
    for channel, group in sorted(by_channel.items()):
        quota = min(len(group), max(min_per_channel, round(total * len(group) / len(rows))))
        picked += rng.sample(group, quota)
    rng.shuffle(picked)  # so batches mix channels
    return picked


def read_eligible(run_id: str) -> list[dict]:
    """Unique eligible Shorts from silver/data_discovery/<run> or silver/eligible_shorts/<run> (csv module only)."""
    candidates = [PROJECT_ROOT / "data" / "silver" / folder / run_id / "eligible_shorts.csv"
                  for folder in ("data_discovery", "eligible_shorts")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Eligible Shorts not found for {run_id}")
    seen: set[str] = set()
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            video_id = row.get("video_id")
            if video_id and video_id not in seen:
                seen.add(video_id)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect automatic captions only; never downloads the video file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-id", nargs="+", help="One or more YouTube video IDs")
    source.add_argument(
        "--source-run-id",
        help="Collect captions for every unique eligible Short in silver/data_discovery/<run_id>/eligible_shorts.csv.",
    )
    source.add_argument(
        "--retry-dlq-from-run-id",
        help="Retry records with --retry-statuses (plus pending ones) from a previous Bronze transcript run.",
    )
    parser.add_argument("--sector", choices=["moda", "automotivo"], help="With --source-run-id: only this sector.")
    parser.add_argument("--languages", nargs="+", default=["pt", "pt-BR", "es", "en"], help="Preferred caption languages")
    parser.add_argument("--provider", choices=["transcript-api", "yt-dlp", "apify"], default="transcript-api")
    parser.add_argument("--apify-actor", default=APIFY_DEFAULT_ACTOR, help="Apify actor id (provider=apify).")
    parser.add_argument("--apify-output-format", default="xmlWithTimestamps",
                        choices=["captions", "textWithTimestamps", "xmlWithoutTimestamps", "xmlWithTimestamps", "singleStringText"])
    parser.add_argument("--apify-fallback-actor", default=APIFY_FALLBACK_ACTOR,
                        help="Per-video actor with explicit language for dubbed tracks; 'none' disables.")
    parser.add_argument("--apify-batch-size", type=int, default=25, help="Videos per Apify actor run.")
    parser.add_argument("--max-cost-usd", type=float, default=2.0,
                        help="provider=apify: refuse to start if the estimated cost exceeds this value.")
    parser.add_argument("--sample", type=int, default=0,
                        help="With --source-run-id: stratified sample of N videos by channel_id (deterministic with --seed).")
    parser.add_argument("--sample-min-per-channel", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Maximum videos to attempt in this run; 0 = all.")
    parser.add_argument("--max-rate-limit-retries", type=int, default=2, help="Retries only for rate_limited; blocked videos are never retried.")
    parser.add_argument("--backoff-base-seconds", type=float, default=15, help="Base delay for exponential backoff.")
    parser.add_argument("--backoff-max-seconds", type=float, default=120, help="Maximum delay for one retry.")
    parser.add_argument("--sleep-between-seconds", type=float, default=10,
                        help="Base pause between videos; actual pause is base..2*base (jitter). 0 disables.")
    parser.add_argument("--max-consecutive-blocked", type=int, default=2,
                        help="Abort the run after N consecutive blocked videos; 0 disables.")
    parser.add_argument("--retry-statuses", nargs="+", default=["rate_limited", "blocked"],
                        choices=["rate_limited", "blocked", "failed", "not_available"])
    parser.add_argument("--no-skip-known", action="store_true",
                        help="Do not skip videos already downloaded in previous runs.")
    parser.add_argument("--recheck-not-available", action="store_true",
                        help="Also re-attempt videos previously marked not_available (they are skipped by default).")
    args = parser.parse_args()
    if min(args.limit, args.max_rate_limit_retries, args.sleep_between_seconds, args.max_consecutive_blocked) < 0 \
            or args.backoff_base_seconds <= 0 or args.backoff_max_seconds <= 0:
        raise ValueError("Limits and pauses must be non-negative; backoff delays must be positive")

    if args.source_run_id:
        eligible = read_eligible(args.source_run_id)
        if args.sector:
            eligible = [r for r in eligible if r.get("sector") == args.sector]
        if args.sample:
            eligible = stratified_sample(eligible, args.sample, args.sample_min_per_channel, args.seed)
            print(f"Stratified sample: {len(eligible)} videos across {len({r.get('channel_id') for r in eligible})} channels (seed {args.seed}).", flush=True)
        video_ids = [r["video_id"] for r in eligible]
    elif args.retry_dlq_from_run_id:
        previous_path = TRANSCRIPT_ROOT / args.retry_dlq_from_run_id / "run_summary.json"
        if not previous_path.exists():
            raise FileNotFoundError(f"Transcript run summary not found: {previous_path}")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        video_ids = [r["video_id"] for r in previous["records"] if r.get("transcript_status") in args.retry_statuses]
        video_ids += previous.get("pending_video_ids", [])
        video_ids = list(dict.fromkeys(video_ids))
    else:
        video_ids = list(dict.fromkeys(args.video_id))

    skipped: dict[str, str] = {}
    if not args.no_skip_known:
        outcomes = known_outcomes(include_not_available=not args.recheck_not_available)
        skipped = {v: outcomes[v] for v in video_ids if v in outcomes}
        video_ids = [v for v in video_ids if v not in outcomes]
    if args.limit:
        video_ids = video_ids[: args.limit]

    apify_token: str | None = None
    if args.provider == "apify":
        token = dotenv_values(PROJECT_ROOT / ".env").get("APIFY_TOKEN")
        if not token:
            raise RuntimeError("Set APIFY_TOKEN in .env before using --provider apify.")
        if args.apify_batch_size < 1:
            raise ValueError("--apify-batch-size must be positive")
        estimate = len(video_ids) * APIFY_DEFAULT_PRICE_PER_VIDEO_USD
        print(f"Apify: {len(video_ids)} videos, estimated cost ~US$ {estimate:.2f} (limit US$ {args.max_cost_usd:.2f}).", flush=True)
        if estimate > args.max_cost_usd:
            raise SystemExit("Estimated cost above --max-cost-usd. Use --limit or raise --max-cost-usd.")
        apify_token = token

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = TRANSCRIPT_ROOT / run_id
    files_dir = bronze_dir / "files"
    silver_dir = PROJECT_ROOT / "data" / "silver" / "transcript_manifest" / run_id
    files_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)

    raw_dir = bronze_dir / "raw"
    api = None if args.provider == "apify" else _transcript_api()  # one shared HTTP session for the whole run
    records: list[dict] = []
    pending: list[str] = list(video_ids)
    summary_path = bronze_dir / "run_summary.json"
    events_path = bronze_dir / "collection_events.jsonl"

    def summary(status: str, aborted_reason: str | None = None) -> dict:
        return {
            "run_id": run_id,
            "status": status,
            "aborted_reason": aborted_reason,
            "provider": args.provider,
            "sleep_between_seconds": args.sleep_between_seconds if args.provider != "apify" else None,
            "apify_actor": args.apify_actor if args.provider == "apify" else None,
            "expected_records": len(video_ids),
            "completed_records": len(records),
            "skipped_known_records": len(skipped),
            "pending_video_ids": pending,
            "records": records,
        }

    print(f"Skipping {len(skipped)} videos already resolved in previous runs; attempting {len(video_ids)}.", flush=True)
    write_json(summary_path, summary("running"))
    append_jsonl(events_path, {"event": "run_started", "run_id": run_id, "occurred_at": now(),
                               "expected_records": len(video_ids), "skipped_known_records": len(skipped)})

    consecutive_blocked = 0
    aborted_reason: str | None = None
    def record_done(record: dict) -> None:
        records.append(record)
        pending.remove(record["video_id"])
        append_jsonl(events_path, {"event": "video_completed", **record})

    try:
        if apify_token is not None:
            raw_dir.mkdir()
            batches = [video_ids[i:i + args.apify_batch_size] for i in range(0, len(video_ids), args.apify_batch_size)]
            for number, batch in enumerate(batches, start=1):
                print(f"[batch {number}/{len(batches)}] sending {len(batch)} videos to Apify...", flush=True)
                try:
                    batch_records = collect_batch_with_apify(
                        apify_token, args.apify_actor, args.apify_output_format, batch,
                        files_dir, raw_dir, args.languages, run_id,
                        None if args.apify_fallback_actor == "none" else args.apify_fallback_actor,
                    )
                except Exception as error:  # auth, credits, actor failure: stop and keep the rest pending
                    aborted_reason = f"apify_error: {type(error).__name__}: {str(error)[:300]}"
                    print(f"ABORTING: {aborted_reason}", flush=True)
                    break
                for record in batch_records:
                    record_done(record)
                write_json(summary_path, summary("running"))
                counts = {s: sum(r["transcript_status"] == s for r in batch_records) for s in ("downloaded", "not_available", "failed")}
                print(f"  {counts}", flush=True)
                if counts["downloaded"] == 0 and counts["failed"] == len(batch_records):
                    aborted_reason = "apify_batch_all_failed"
                    print("ABORTING: every video in the batch failed; check the raw/ output.", flush=True)
                    break
        for index, video_id in enumerate(video_ids if apify_token is None else []):
            if index and args.sleep_between_seconds:
                time.sleep(args.sleep_between_seconds + random.uniform(0, args.sleep_between_seconds))
            print(f"[{index + 1}/{len(video_ids)}] captions only for {video_id}...", flush=True)
            record = collect_with_rate_limit_retry(
                api, video_id, files_dir, args.languages, run_id, args.provider,
                args.max_rate_limit_retries, args.backoff_base_seconds, args.backoff_max_seconds,
            )
            record_done(record)
            write_json(summary_path, summary("running"))
            print(f"  {record['transcript_status']} ({record.get('error_code') or 'ok'})", flush=True)

            consecutive_blocked = consecutive_blocked + 1 if record["transcript_status"] == "blocked" else 0
            if args.max_consecutive_blocked and consecutive_blocked >= args.max_consecutive_blocked:
                aborted_reason = "consecutive_blocked"
                print(f"ABORTING: {consecutive_blocked} consecutive blocks. Your IP is likely banned; "
                      "wait several hours before resuming.", flush=True)
                break
    except KeyboardInterrupt:
        aborted_reason = "interrupted"
        print("Interrupted; saving progress.", flush=True)

    status = "aborted" if aborted_reason else "completed"
    write_json(silver_dir / "transcript_manifest.json", records)
    write_json(summary_path, summary(status, aborted_reason))
    for dlq_status in ("rate_limited", "blocked"):
        write_json(bronze_dir / f"{dlq_status}_dlq.json", {
            "run_id": run_id, "created_at": now(),
            "records": [r for r in records if r["transcript_status"] == dlq_status],
        })
    downloaded = sum(r["transcript_status"] == "downloaded" for r in records)
    append_jsonl(events_path, {"event": f"run_{status}", "run_id": run_id, "occurred_at": now(),
                               "aborted_reason": aborted_reason, "downloaded_records": downloaded,
                               "total_records": len(records), "pending_records": len(pending)})
    print(f"Run {run_id} {status}. Downloaded: {downloaded}/{len(records)}. Pending: {len(pending)}.")


if __name__ == "__main__":
    main()
