"""Rubrica do video (secao 6.2 do TAP) para um criativo: cache da Silver ou Gemini ao vivo.

Ordem de busca:
1. Silver `data/silver/gemini_video/gemini_video_features.csv` (Shorts da base, coletados em lote);
2. cache local de chamadas ao vivo (`.cache/gemini_rubric/<video_id>.json`);
3. Gemini ao vivo a partir da URL publica do YouTube (mesmo prompt, schema e modelo da coleta em lote,
   importados de scripts/coleta/collect_gemini_video_features.py). Exige GEMINI_API_KEY;
4. sem rubrica -> a tool usa o modelo sem rubrica (CB_EMB) e marca confianca baixa.

O video nunca e baixado: o Gemini recebe so a URL publica.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _api_key(root: Path) -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    env = root / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


class VideoRubric:
    def __init__(self, root: Path, extra_csv: Optional[list[Path]] = None, live: Optional[bool] = None,
                 model: Optional[str] = None, timeout: int = 180):
        self.root = Path(root)
        self.cache_dir = self.root / ".cache" / "gemini_rubric"
        self.model = model or os.environ.get("GALAXIES_RUBRIC_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        # live=None: chama o Gemini se houver chave; GALAXIES_RUBRIC_LIVE=0 desliga (ex.: testes offline)
        self.live = live if live is not None else os.environ.get("GALAXIES_RUBRIC_LIVE", "1") != "0"
        self._rows: dict[str, dict] = {}
        for path in [self.root / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"] + list(extra_csv or []):
            if Path(path).exists():
                with open(path, encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        self._rows[row["video_id"]] = row
        self._collector = None

    def _load_collector(self):
        if self._collector is None:
            path = self.root / "scripts" / "coleta" / "collect_gemini_video_features.py"
            spec = importlib.util.spec_from_file_location("_gemini_collector", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._collector = module
        return self._collector

    def get(self, video_id: Optional[str], sector: str) -> tuple[Optional[dict], str]:
        """Devolve (rubrica, fonte). fonte: cache | gemini_ao_vivo | motivo da ausencia."""
        if not video_id:
            return None, "sem_video (criativo nao e um Short do YouTube)"
        if video_id in self._rows:
            return self._rows[video_id], "cache"
        cached = self.cache_dir / f"{video_id}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8")), "cache"
        if not self.live:
            return None, "gemini_desligado"
        key = _api_key(self.root)
        if not key:
            return None, "sem_chave_gemini"
        try:
            rubric = self._call(key, video_id, sector)
        except Exception as error:  # rede, cota, video privado: a tool segue sem rubrica
            return None, f"gemini_falhou ({type(error).__name__}: {str(error)[:120]})"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rubric, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, cached)
        self._rows[video_id] = rubric
        return rubric, "gemini_ao_vivo"

    def _call(self, key: str, video_id: str, sector: str) -> dict:
        c = self._load_collector()
        body = {
            "contents": [{"role": "user", "parts": [
                {"file_data": {"file_uri": f"https://www.youtube.com/watch?v={video_id}"}},
                {"text": c.PROMPT.format(sector=sector, sector_hint=c.SECTOR_HINTS.get(sector, ""))},
            ]}],
            "generationConfig": {"response_mime_type": "application/json", "response_schema": c.SCHEMA,
                                 "temperature": 0, "media_resolution": "MEDIA_RESOLUTION_LOW"},
        }
        req = urllib.request.Request(f"{c.API}/models/{self.model}:generateContent",
                                     data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}") from None
        result = c.parse_response(payload)
        problems = c.validate(result)
        if problems:
            raise ValueError("; ".join(problems[:3]))
        for list_key in ("elementos_video_top5", "elementos_audio_top5"):
            if isinstance(result.get(list_key), list):
                result[list_key] = " | ".join(str(v) for v in result[list_key])
        return {k: (int(v) if isinstance(v, bool) else v) for k, v in result.items()}
