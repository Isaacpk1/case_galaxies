"""Encontra o channel_id de canais candidatos pela busca da YouTube API (so biblioteca padrao).

Para cada linha de data/candidates/channel_queries_v2.csv (channel_name, sector,
channel_type, query) faz search.list (100 unidades) e channels.list (1 unidade)
e grava os 3 primeiros resultados com inscritos e numero de videos, para
revisao humana antes de entrar em channels_v2.csv.

Uso (PowerShell):
  python scripts/coleta/resolve_channels_stdlib.py
Saida: data/candidates/resolved_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import collect_youtube_stdlib as yt

PROJECT_ROOT = yt.PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "candidates" / "channel_queries_v2.csv")
    parser.add_argument("--hits", type=int, default=3)
    args = parser.parse_args()
    if not yt.API_KEY:
        raise SystemExit("Defina YOUTUBE_API_KEY no .env antes de rodar.")
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        queries = list(csv.DictReader(stream))
    out_rows = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q['query']}...", end=" ", flush=True)
        try:
            hits = yt.youtube_get("search", part="snippet", q=q["query"], type="channel",
                                  regionCode="BR", relevanceLanguage="pt", maxResults=args.hits).get("items", [])
            ids = [h["snippet"]["channelId"] for h in hits]
            details = {c["id"]: c for c in yt.youtube_get("channels", part="snippet,statistics",
                                                           id=",".join(ids)).get("items", [])} if ids else {}
            for rank, cid in enumerate(ids, 1):
                c = details.get(cid, {})
                out_rows.append({**q, "rank": rank, "channel_id": cid,
                                 "title": c.get("snippet", {}).get("title", ""),
                                 "handle": c.get("snippet", {}).get("customUrl", ""),
                                 "country": c.get("snippet", {}).get("country", ""),
                                 "subscribers": c.get("statistics", {}).get("subscriberCount", ""),
                                 "videos": c.get("statistics", {}).get("videoCount", "")})
            print(", ".join(out_rows[-k]["title"] for k in range(len(ids), 0, -1)) if ids else "nada")
        except RuntimeError as error:
            print(f"ERRO: {error}")
    out = PROJECT_ROOT / "data" / "candidates" / f"resolved_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    if out_rows:
        with out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    print(f"\n{len(out_rows)} resultados em {out}. Cota usada: ~{yt.QUOTA['units']} unidades.")


if __name__ == "__main__":
    main()
