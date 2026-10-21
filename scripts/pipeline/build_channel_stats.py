"""Silver de estatisticas dos canais (inscritos e videos) a partir do Bronze do YouTube (so biblioteca padrao).

O coletor (collect_youtube_stdlib.py) ja grava, em cada run, `channels.json` com `subscriber_count_at_collection`
e `video_count_at_collection` (channels.list, part=statistics). Este script junta todos os runs e fica com o
registro MAIS RECENTE de cada canal. Nao chama a API.

Atencao (vazamento): o numero de inscritos e uma fotografia da data da coleta, nao da data de cada post; por
isso ele e usado so no baseline declarado do TAP (regressao linear sobre inscritos) e como informacao no
contrato da tool, nunca como feature do modelo treinado.

Saida: data/silver/channel_stats/<timestamp>/channel_stats.csv + manifest.json

    python scripts/pipeline/build_channel_stats.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze" / "youtube"
OUT_BASE = ROOT / "data" / "silver" / "channel_stats"
FIELDS = ["channel_id", "channel_name", "sector", "channel_type", "subscriber_count_at_collection",
          "video_count_at_collection", "hidden_subscriber_count", "collected_at", "bronze_run_id"]


def records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("channels", data.get("items", []))
    return [r for r in items if isinstance(r, dict) and r.get("channel_id")]


def main() -> None:
    latest: dict[str, dict] = {}
    runs = sorted(p.parent.name for p in BRONZE.glob("*/channels.json"))
    for run in runs:
        for rec in records(BRONZE / run / "channels.json"):
            row = {k: rec.get(k, "") for k in FIELDS}
            row["bronze_run_id"] = run
            subs = rec.get("subscriber_count_at_collection")
            row["hidden_subscriber_count"] = subs in (None, "")
            previous = latest.get(rec["channel_id"])
            if previous is None or str(row["collected_at"]) >= str(previous["collected_at"]):
                latest[rec["channel_id"]] = row
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_BASE / version
    out.mkdir(parents=True, exist_ok=False)
    with (out / "channel_stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(latest.values(), key=lambda r: r["channel_id"]))
    manifest = {"version": version, "created_at": datetime.now(timezone.utc).isoformat(), "bronze_runs": runs,
                "channels": len(latest), "hidden_subscriber_count": sum(r["hidden_subscriber_count"] for r in latest.values()),
                "regra": "registro mais recente de cada canal entre todos os runs do Bronze",
                "aviso": "inscritos na data da coleta (fotografia), nao na data de cada post"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
