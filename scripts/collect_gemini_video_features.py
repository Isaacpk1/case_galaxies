"""Coleta atributos de video (rubrica VLM do TAP) e transcricao com o Gemini a partir da URL publica do YouTube.

O video nao e baixado: a API do Gemini recebe a URL publica do Short e devolve um JSON fixo com
a rubrica da secao 6.2 do TAP (gancho, rosto, produto, texto na tela, ritmo, audio, estetica) e a
transcricao da fala. Somente biblioteca padrao (roda no Windows do Inteli, no Colab ou no Linux).

Bronze imutavel por run:
    data/bronze/gemini_video/<run_id>/responses/<video_id>.json   resposta bruta da API
    data/bronze/gemini_video/<run_id>/manifest.jsonl              uma linha por tentativa
    data/bronze/gemini_video/<run_id>/run_config.json             parametros, prompt e hash

Regras:
- `video_id` ja resolvido (status ok) em qualquer run anterior e pulado.
- Limite diario do plano gratuito para URL do YouTube (8 h de video/dia, conforme a documentacao em 10/2026):
  o coletor para ao atingir `--max-video-hours` (padrao 7,5 h) somando `duration_seconds`.
- 429/5xx: backoff exponencial com jitter, respeitando Retry-After; 5 falhas seguidas abrem o circuit breaker.
- A chave vem de GEMINI_API_KEY (ambiente ou .env) e nunca e gravada.

Exemplos:
    python scripts/collect_gemini_video_features.py --list-models
    python scripts/collect_gemini_video_features.py --limit 100 --dry-run
    python scripts/collect_gemini_video_features.py --limit 100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze" / "gemini_video"
API = "https://generativelanguage.googleapis.com/v1beta"
EXTRACTOR_VERSION = "1.0.0"
PROMPT_VERSION = "rubrica_tap_v1"

PROMPT = """Voce e um analista de criativos de video curto (YouTube Shorts) do mercado brasileiro.
Assista ao video inteiro, com audio, e preencha a rubrica em JSON. Regras:
- Avalie SOMENTE o conteudo do video (imagem, texto na tela e audio). Ignore titulo, descricao,
  numero de visualizacoes, curtidas, comentarios, nome ou fama do canal: a rubrica sera usada para
  prever desempenho ANTES da publicacao.
- Notas *_0a10 sao inteiros de 0 a 10, onde 5 e um Short tipico do setor.
- hook_primeiros_3s: descreva em ate 15 palavras o que acontece nos primeiros 3 segundos.
- transcricao: transcreva a fala literalmente, no idioma original, sem traduzir. Se nao houver fala,
  use string vazia. Nao transcreva letras de musica; marque apenas que ha musica.
- idioma_fala: codigo ISO 639-1 (ex.: pt, en, es) ou "nenhum".
- elementos_video_top5 / elementos_audio_top5: descritores curtos e padronizados, em portugues, minusculos
  (ex.: "close do produto", "pessoa falando para a camera", "musica eletronica", "ronco do motor").
- Setor {sector}: {sector_hint}
"""

SECTOR_HINTS = {
    "moda": "observe try-on (pessoa vestindo a peca), corpo inteiro vs. detalhe da peca, styling, haul, antes/depois de look.",
    "automotivo": "observe close do carro, interior, POV de direcao, som do motor, test drive, comparacao de modelos.",
    "beleza": "observe swatch, aplicacao na pele, resultado antes/depois, close no rosto.",
}

SCORE = {"type": "INTEGER", "minimum": 0, "maximum": 10}
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hook_primeiros_3s": {"type": "STRING"},
        "forca_hook_0a10": SCORE,
        "tem_rosto": {"type": "BOOLEAN"},
        "pessoa_fala_para_camera": {"type": "BOOLEAN"},
        "close_up_produto": {"type": "BOOLEAN"},
        "before_after_ou_transformacao": {"type": "BOOLEAN"},
        "try_on_ou_demonstracao": {"type": "BOOLEAN"},
        "texto_na_tela": {"type": "BOOLEAN"},
        "densidade_texto_0a10": SCORE,
        "ritmo_cortes_0a10": SCORE,
        "tem_musica": {"type": "BOOLEAN"},
        "energia_musica_0a10": SCORE,
        "tom_narracao": {"type": "STRING", "enum": ["ausente", "calmo", "animado", "instrucional"]},
        "chamada_para_acao": {"type": "BOOLEAN"},
        "brilho_saturacao_0a10": SCORE,
        "cor_dominante": {"type": "STRING"},
        "estetica_geral_0a10": SCORE,
        "clareza_beneficio_produto_0a10": SCORE,
        "elementos_video_top5": {"type": "ARRAY", "items": {"type": "STRING"}},
        "elementos_audio_top5": {"type": "ARRAY", "items": {"type": "STRING"}},
        "potencial_viral_0a10": SCORE,
        "idioma_fala": {"type": "STRING"},
        "transcricao": {"type": "STRING"},
    },
    "required": [
        "hook_primeiros_3s", "forca_hook_0a10", "tem_rosto", "pessoa_fala_para_camera", "close_up_produto",
        "before_after_ou_transformacao", "try_on_ou_demonstracao", "texto_na_tela", "densidade_texto_0a10",
        "ritmo_cortes_0a10", "tem_musica", "energia_musica_0a10", "tom_narracao", "chamada_para_acao",
        "brilho_saturacao_0a10", "cor_dominante", "estetica_geral_0a10", "clareza_beneficio_produto_0a10",
        "elementos_video_top5", "elementos_audio_top5", "potencial_viral_0a10", "idioma_fala", "transcricao",
    ],
}


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("GEMINI_API_KEY ausente: adicione ao .env (veja .env.example).")
    return key


def request_json(url: str, key: str, body: dict | None = None, timeout: int = 300) -> tuple[int, dict, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw[:2000]}
        return error.code, payload, dict(error.headers or {})


def list_models(key: str) -> None:
    status, payload, _ = request_json(f"{API}/models?pageSize=200", key)
    if status != 200:
        sys.exit(f"HTTP {status}: {json.dumps(payload)[:500]}")
    for model in payload.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"].removeprefix("models/"), "-", model.get("displayName", ""))


def resolve_source(path: Path | None) -> Path:
    if path:
        return path
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    return ROOT / latest["path"]


def already_done() -> set[str]:
    done: set[str] = set()
    for manifest in BRONZE.glob("*/manifest.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(row["video_id"])
    return done


def stratified_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Amostra equilibrada por setor x tipo de canal, espalhada entre canais."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["sector"], row["channel_type"])].append(row)
    rng = random.Random(seed)
    per_group = max(1, limit // max(1, len(groups)))
    chosen: list[dict] = []
    for key in sorted(groups):
        by_channel: dict[str, list[dict]] = defaultdict(list)
        for row in groups[key]:
            by_channel[row["channel_id"]].append(row)
        for items in by_channel.values():
            rng.shuffle(items)
        channels = sorted(by_channel)
        rng.shuffle(channels)
        picked, i = [], 0
        while len(picked) < per_group and any(by_channel.values()):
            items = by_channel[channels[i % len(channels)]]
            if items:
                picked.append(items.pop())
            i += 1
        chosen.extend(picked)
    return chosen[:limit]


def validate(result: dict) -> list[str]:
    problems = [f"faltando {k}" for k in SCHEMA["required"] if k not in result]
    for key, spec in SCHEMA["properties"].items():
        if key in result and spec.get("type") == "INTEGER" and not (0 <= int(result[key]) <= 10):
            problems.append(f"{key} fora de 0-10")
    return problems


def call_gemini(key: str, model: str, video_id: str, sector: str, media_resolution: str) -> tuple[str, dict, dict]:
    body = {
        "contents": [{"role": "user", "parts": [
            {"file_data": {"file_uri": f"https://www.youtube.com/watch?v={video_id}"}},
            {"text": PROMPT.format(sector=sector, sector_hint=SECTOR_HINTS.get(sector, ""))},
        ]}],
        "generationConfig": {"response_mime_type": "application/json", "response_schema": SCHEMA,
                             "temperature": 0, "media_resolution": media_resolution},
    }
    delay = 5.0
    for attempt in range(6):
        status, payload, headers = request_json(f"{API}/models/{model}:generateContent", key, body)
        if status == 200:
            return "ok", payload, {"http_status": status, "attempts": attempt + 1}
        message = json.dumps(payload)[:800]
        if status in (429, 500, 503) and attempt < 5:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay * (1 + random.random())
            print(f"  {video_id}: HTTP {status}, nova tentativa em {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        kind = {429: "rate_limited", 403: "blocked", 404: "not_available", 400: "failed"}.get(status, "failed")
        if status == 400 and ("private" in message.lower() or "not available" in message.lower()):
            kind = "not_available"
        return kind, payload, {"http_status": status, "attempts": attempt + 1, "error": message}
    return "rate_limited", {}, {"attempts": 6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-csv", type=Path, help="CSV com video_id, channel_id, sector, channel_type, duration_seconds (padrao: Gold LATEST)")
    parser.add_argument("--video-ids", nargs="*", help="Lista explicita de video_id (ignora a amostragem)")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="Todos os pendentes, sem amostragem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--media-resolution", default="MEDIA_RESOLUTION_LOW",
                        choices=["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"])
    parser.add_argument("--rpm", type=float, default=8.0, help="Requisicoes por minuto (respeite a cota do seu plano)")
    parser.add_argument("--max-video-hours", type=float, default=7.5, help="Teto de horas de video por execucao")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_models(load_api_key())
        return

    source = resolve_source(args.source_csv)
    rows = list(csv.DictReader(open(source, encoding="utf-8")))
    done = already_done()
    pending = [r for r in rows if r["video_id"] not in done]
    if args.video_ids:
        wanted = set(args.video_ids)
        batch = [r for r in pending if r["video_id"] in wanted]
    elif args.all:
        batch = pending
    else:
        batch = stratified_sample(pending, args.limit, args.seed)

    seconds = sum(float(r.get("duration_seconds") or 0) for r in batch)
    mix = defaultdict(int)
    for r in batch:
        mix[(r["sector"], r["channel_type"])] += 1
    print(f"fonte: {source}\nja resolvidos: {len(done)} | pendentes: {len(pending)} | lote: {len(batch)}")
    print("mistura:", {f"{a}/{b}": n for (a, b), n in sorted(mix.items())})
    print(f"video total: {seconds / 3600:.2f} h | tokens de entrada estimados (~100/s em resolucao baixa): {seconds * 100:,.0f}")
    if args.dry_run or not batch:
        return

    key = load_api_key()
    run_dir = BRONZE / args.run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=False)
    prompt_hash = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": args.run_id, "model": args.model, "media_resolution": args.media_resolution,
        "extractor_version": EXTRACTOR_VERSION, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash,
        "prompt": PROMPT, "sector_hints": SECTOR_HINTS, "schema": SCHEMA, "source_csv": str(source),
        "video_ids": [r["video_id"] for r in batch], "usage_note": "URL publica do YouTube enviada ao Gemini; video nao baixado.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    interval = 60.0 / args.rpm
    used_seconds, consecutive_failures = 0.0, 0
    counts = defaultdict(int)
    manifest = open(run_dir / "manifest.jsonl", "a", encoding="utf-8")
    for n, row in enumerate(batch, 1):
        duration = float(row.get("duration_seconds") or 0)
        if (used_seconds + duration) / 3600 > args.max_video_hours:
            print(f"Teto de {args.max_video_hours} h de video atingido; retome depois com o mesmo comando.")
            break
        started = time.time()
        status, payload, meta = call_gemini(key, args.model, row["video_id"], row["sector"], args.media_resolution)
        record = {"video_id": row["video_id"], "channel_id": row["channel_id"], "sector": row["sector"],
                  "run_id": args.run_id, "model": args.model, "prompt_version": PROMPT_VERSION,
                  "collected_at": datetime.now(timezone.utc).isoformat(), "latency_s": round(time.time() - started, 1),
                  "duration_seconds": duration, **meta}
        if status == "ok":
            try:
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text)
                problems = validate(result)
                record["status"] = "invalid_json" if problems else "ok"
                if problems:
                    record["error"] = "; ".join(problems)
            except (KeyError, IndexError, json.JSONDecodeError) as error:
                record["status"], record["error"] = "invalid_json", repr(error)
                finish = (payload.get("candidates") or [{}])[0].get("finishReason")
                if finish:
                    record["finish_reason"] = finish
            record["usage"] = payload.get("usageMetadata", {})
            (run_dir / "responses" / f"{row['video_id']}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        else:
            record["status"] = status
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest.flush()
        counts[record["status"]] += 1
        used_seconds += duration
        consecutive_failures = 0 if record["status"] == "ok" else consecutive_failures + 1
        print(f"[{n}/{len(batch)}] {row['video_id']} {record['status']} ({record['latency_s']}s)", flush=True)
        if consecutive_failures >= 5:
            print("Circuit breaker: 5 falhas seguidas. Veja o manifest antes de continuar.")
            break
        time.sleep(max(0.0, interval - (time.time() - started)))
    manifest.close()

    tokens = defaultdict(int)
    for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        for k, v in json.loads(line).get("usage", {}).items():
            if isinstance(v, int):
                tokens[k] += v
    print("status:", dict(counts))
    print("tokens:", dict(tokens))
    print("saida:", run_dir)


if __name__ == "__main__":
    main()
