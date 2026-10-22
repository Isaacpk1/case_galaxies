"""Encontra e tria canais candidatos pela YouTube API (so biblioteca padrao).

Para cada linha de data/candidates/channel_queries_v2.csv (channel_name, sector,
channel_type, handle, query):
  1. tenta o handle (channels.list forHandle, 1 unidade);
  2. se nao achar, usa a busca (search.list, 100 unidades) e fica com o 1o resultado;
  3. triagem: le os 50 uploads mais recentes (2 unidades) e conta quantos sao Shorts
     (1-180 s) publicados na janela de 18 meses e com >= 30 dias.
Grava data/candidates/resolved_<timestamp>.csv para revisao antes da coleta completa.

Uso (PowerShell, na raiz do repo):
  python scripts/coleta/resolve_channels_stdlib.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import collect_youtube_stdlib as yt

PROJECT_ROOT = yt.PROJECT_ROOT


def find_channel(row: dict) -> tuple[dict | None, str]:
    handle = (row.get("handle") or "").strip()
    if handle:
        items = yt.youtube_get("channels", part="snippet,statistics,contentDetails",
                               forHandle=handle if handle.startswith("@") else "@" + handle).get("items", [])
        if items:
            return items[0], "handle"
    query = (row.get("query") or row.get("channel_name") or "").strip()
    if not query:
        return None, "sem_handle_nem_query"
    hits = yt.youtube_get("search", part="snippet", q=query, type="channel", regionCode="BR",
                          relevanceLanguage="pt", maxResults=1).get("items", [])
    if not hits:
        return None, "busca_sem_resultado"
    items = yt.youtube_get("channels", part="snippet,statistics,contentDetails",
                           id=hits[0]["snippet"]["channelId"]).get("items", [])
    return (items[0], "busca") if items else (None, "busca_sem_detalhe")


def screen(channel: dict, history_start: datetime, cutoff: datetime) -> dict:
    uploads = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.youtube_get("playlistItems", part="contentDetails", playlistId=uploads, maxResults=50).get("items", [])
    ids = [i["contentDetails"]["videoId"] for i in items]
    if not ids:
        return {"recent_uploads": 0, "recent_shorts_in_window": 0, "last_upload": ""}
    videos = yt.youtube_get("videos", part="snippet,contentDetails", id=",".join(ids)).get("items", [])
    shorts = 0
    for v in videos:
        published = yt.parse_time(v["snippet"].get("publishedAt"))
        if (published and history_start <= published <= cutoff
                and 1 <= yt.duration_seconds(v["contentDetails"].get("duration", "")) <= yt.MAX_SHORT_DURATION_SECONDS):
            shorts += 1
    last = max((v["snippet"].get("publishedAt", "") for v in videos), default="")
    return {"recent_uploads": len(videos), "recent_shorts_in_window": shorts, "last_upload": last[:10]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "candidates" / "channel_queries_v2.csv")
    args = parser.parse_args()
    if not yt.API_KEY:
        raise SystemExit("Defina YOUTUBE_API_KEY no .env antes de rodar.")
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    now = datetime.now(timezone.utc)
    history_start, cutoff = now - timedelta(days=yt.HISTORY_DAYS), now - timedelta(days=yt.MIN_POST_AGE_DAYS)
    out_rows = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['channel_name']}...", end=" ", flush=True)
        record = {**row, "found_by": "", "channel_id": "", "title": "", "found_handle": "", "country": "",
                  "subscribers": "", "videos": "", "recent_uploads": "", "recent_shorts_in_window": "", "last_upload": ""}
        try:
            channel, how = find_channel(row)
            record["found_by"] = how
            if channel:
                sn, st = channel.get("snippet", {}), channel.get("statistics", {})
                record.update({"channel_id": channel["id"], "title": sn.get("title", ""),
                               "found_handle": sn.get("customUrl", ""), "country": sn.get("country", ""),
                               "subscribers": st.get("subscriberCount", ""), "videos": st.get("videoCount", "")})
                record.update(screen(channel, history_start, cutoff))
                print(f"{record['title']} ({how}) - {record['recent_shorts_in_window']}/50 Shorts recentes na janela")
            else:
                print(f"nao encontrado ({how})")
        except RuntimeError as error:
            record["found_by"] = f"erro: {error}"
            print(f"ERRO: {error}")
        out_rows.append(record)
    out = PROJECT_ROOT / "data" / "candidates" / f"resolved_{now:%Y%m%dT%H%M%SZ}.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    found = [r for r in out_rows if r["channel_id"]]
    promising = [r for r in found if int(r["recent_shorts_in_window"] or 0) >= 10]
    print(f"\nEncontrados {len(found)}/{len(rows)}; com >= 10 Shorts recentes na janela: {len(promising)}.")
    print(f"Arquivo: {out}. Cota usada: ~{yt.QUOTA['units']} unidades.")


if __name__ == "__main__":
    main()
