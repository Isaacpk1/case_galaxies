"""Coleta atributos de video (rubrica VLM do TAP) e transcricao com o Gemini a partir da URL publica do YouTube.

O video nao e baixado: a API do Gemini recebe a URL publica do Short e devolve um JSON fixo com
a rubrica da secao 6.2 do TAP (gancho, rosto, produto, texto na tela, ritmo, audio, estetica) e a
transcricao da fala. Somente biblioteca padrao (roda no Windows do Inteli, no Colab ou no Linux).

Bronze imutavel por run:
    data/bronze/gemini_video/<run_id>/responses/<video_id>.json   resposta bruta da API
    data/bronze/gemini_video/<run_id>/manifest.jsonl              uma linha por tentativa
    data/bronze/gemini_video/<run_id>/run_config.json             parametros, prompt e hash
Silver consolidada (uma linha por video_id, sem duplicados):
    data/silver/gemini_video/gemini_video_features.csv
    data/silver/gemini_video/consolidation_report.json

Garantias para rodar em varias sessoes ate acabar a cota:
- Cada resposta e gravada no disco assim que chega (arquivo temporario + rename atomico) e o manifest
  e sincronizado a cada video. Queda de energia, Ctrl+C ou erro perdem no maximo o video em andamento.
- Nenhum video e analisado duas vezes: um video conta como resolvido se tiver status ok em qualquer
  manifest OU uma resposta valida salva em qualquer run (cobre queda entre resposta e manifest).
  Videos `not_available` tambem sao pulados (use --retry-unavailable para tentar de novo).
- Um arquivo de trava impede dois coletores rodando ao mesmo tempo.
- Cota diaria do plano gratuito para URL do YouTube (8 h de video/dia, documentacao em 10/2026): o
  orcamento desconta o video ja enviado nas ultimas 24 h em todos os runs. 429 de cota diaria encerra
  o run na hora (sem insistir); 429/5xx transitorios usam backoff com Retry-After.
- Ao final de cada execucao (inclusive interrompida) a Silver consolidada e regravada.
- A chave vem de GEMINI_API_KEY (ambiente ou .env) e nunca e gravada.

Exemplos:
    python scripts/collect_gemini_video_features.py --list-models
    python scripts/collect_gemini_video_features.py --all --dry-run
    python scripts/collect_gemini_video_features.py --all            # repetir ate acabar; retoma sozinho
    python scripts/collect_gemini_video_features.py --consolidate    # so regrava a Silver
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze" / "gemini_video"
SILVER = ROOT / "data" / "silver" / "gemini_video"
LOCK = BRONZE / ".collector.lock"
API = "https://generativelanguage.googleapis.com/v1beta"
EXTRACTOR_VERSION = "1.1.0"
PROMPT_VERSION = "rubrica_tap_v1"


class DailyQuotaExceeded(Exception):
    """Cota diaria atingida: parar o run sem insistir."""


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


# ---------------------------------------------------------------- utilidades

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


def write_atomic(path: Path, text: str) -> None:
    """Grava em arquivo temporario e renomeia: nunca deixa um JSON pela metade."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_response(payload: dict) -> dict:
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def validate(result: dict) -> list[str]:
    problems = [f"faltando {k}" for k in SCHEMA["required"] if k not in result]
    for key, spec in SCHEMA["properties"].items():
        if key in result and spec.get("type") == "INTEGER":
            try:
                if not 0 <= int(result[key]) <= 10:
                    problems.append(f"{key} fora de 0-10")
            except (TypeError, ValueError):
                problems.append(f"{key} nao inteiro")
    return problems


def read_manifests() -> list[dict]:
    rows = []
    for manifest in sorted(BRONZE.glob("*/manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # linha cortada por queda: ignora, o video sera revisto pela resposta salva
    return rows


def valid_saved_responses() -> dict[str, Path]:
    """video_id -> resposta valida mais recente salva em qualquer run."""
    found: dict[str, Path] = {}
    for path in sorted(BRONZE.glob("*/responses/*.json")):
        try:
            result = parse_response(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
        if not validate(result):
            found[path.stem] = path  # sorted por run_id: o mais recente sobrescreve
    return found


def resolved_ids(retry_unavailable: bool) -> tuple[set[str], set[str]]:
    manifests = read_manifests()
    done = {r["video_id"] for r in manifests if r.get("status") == "ok"} | set(valid_saved_responses())
    unavailable = set() if retry_unavailable else {r["video_id"] for r in manifests if r.get("status") == "not_available"}
    return done, unavailable - done


def seconds_sent_last_24h() -> float:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    total = 0.0
    for row in read_manifests():
        try:
            when = datetime.fromisoformat(row["collected_at"])
        except (KeyError, ValueError):
            continue
        if when >= since and row.get("status") in ("ok", "invalid_json"):
            total += float(row.get("duration_seconds") or 0)
    return total


class Lock:
    def __enter__(self):
        BRONZE.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = LOCK.read_text(encoding="utf-8", errors="replace") if LOCK.exists() else ""
            sys.exit(f"Outro coletor parece estar rodando ({info.strip()}). Se nao estiver, apague {LOCK}.")
        os.write(self.fd, f"pid={os.getpid()} inicio={datetime.now(timezone.utc).isoformat()}".encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        LOCK.unlink(missing_ok=True)


# ---------------------------------------------------------------- API

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
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        return 0, {"network_error": repr(error)}, {}


def list_models(key: str) -> None:
    status, payload, _ = request_json(f"{API}/models?pageSize=200", key)
    if status != 200:
        sys.exit(f"HTTP {status}: {json.dumps(payload)[:500]}")
    for model in payload.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"].removeprefix("models/"), "-", model.get("displayName", ""))


def is_daily_quota(payload: dict) -> bool:
    text = json.dumps(payload).lower()
    return "perday" in text or "per_day" in text or "daily" in text or "per day" in text


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
        if status == 429 and is_daily_quota(payload):
            raise DailyQuotaExceeded(message)
        if status in (0, 429, 500, 502, 503, 504) and attempt < 5:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay * (1 + random.random())
            print(f"  {video_id}: HTTP {status}, nova tentativa em {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        kind = {0: "failed", 429: "rate_limited", 403: "blocked", 404: "not_available"}.get(status, "failed")
        if status == 400 and any(t in message.lower() for t in ("private", "not available", "unavailable", "not found")):
            kind = "not_available"
        return kind, payload, {"http_status": status, "attempts": attempt + 1, "error": message}
    return "rate_limited", {}, {"attempts": 6}


# ---------------------------------------------------------------- Silver

FLAT_LISTS = ("elementos_video_top5", "elementos_audio_top5")


def consolidate(source: Path | None = None) -> Path:
    """Uma linha por video_id com a resposta valida mais recente; nunca duplica."""
    SILVER.mkdir(parents=True, exist_ok=True)
    manifests = read_manifests()
    meta: dict[str, dict] = {}
    for row in manifests:  # ordenado por run: o ultimo ok de cada video vence
        if row.get("status") == "ok":
            meta[row["video_id"]] = row
    responses = valid_saved_responses()
    out_rows = []
    for video_id, path in sorted(responses.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = parse_response(payload)
        row = {"video_id": video_id, "gemini_run_id": path.parent.parent.name,
               "model": meta.get(video_id, {}).get("model", ""), "prompt_version": PROMPT_VERSION,
               "prompt_tokens": payload.get("usageMetadata", {}).get("promptTokenCount", "")}
        for key in SCHEMA["required"]:
            value = result.get(key)
            if key in FLAT_LISTS:
                value = " | ".join(str(v) for v in (value or []))
            elif isinstance(value, bool):
                value = int(value)
            row[key] = value
        out_rows.append(row)
    target = SILVER / "gemini_video_features.csv"
    tmp = target.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        fields = ["video_id", "gemini_run_id", "model", "prompt_version", "prompt_tokens"] + SCHEMA["required"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp, target)

    statuses = defaultdict(int)
    last_status: dict[str, str] = {}
    for row in manifests:
        last_status[row["video_id"]] = row.get("status", "")
    for video_id in responses:
        last_status[video_id] = "ok"  # inclui respostas salvas sem linha no manifest (queda)
    for status in last_status.values():
        statuses[status] += 1
    report = {"updated_at": datetime.now(timezone.utc).isoformat(), "videos_with_features": len(out_rows),
              "videos_attempted": len(last_status), "last_status_by_video": dict(statuses),
              "duplicate_attempts_ok": sum(1 for r in manifests if r.get("status") == "ok") - len({r["video_id"] for r in manifests if r.get("status") == "ok"})}
    if source and source.exists():
        total = len({r["video_id"] for r in csv.DictReader(open(source, encoding="utf-8"))})
        report["source_rows"] = total
        report["coverage"] = round(len(out_rows) / total, 4) if total else 0
    write_atomic(SILVER / "consolidation_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Silver: {len(out_rows)} videos em {target}" + (f" (cobertura {report['coverage']:.1%})" if "coverage" in report else ""))
    return target


# ---------------------------------------------------------------- execucao

def resolve_source(path: Path | None) -> Path:
    if path:
        return path
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    return ROOT / latest["path"]


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


def interleave_by_channel(rows: list[dict], seed: int) -> list[dict]:
    """Ordem de coleta alternando canais: se a cota acabar no meio, a cobertura parcial fica equilibrada."""
    rng = random.Random(seed)
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_channel[row["channel_id"]].append(row)
    for items in by_channel.values():
        rng.shuffle(items)
    ordered, channels = [], sorted(by_channel)
    while any(by_channel.values()):
        for channel in channels:
            if by_channel[channel]:
                ordered.append(by_channel[channel].pop())
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-csv", type=Path, help="CSV com video_id, channel_id, sector, channel_type, duration_seconds (padrao: Gold LATEST)")
    parser.add_argument("--video-ids", nargs="*", help="Lista explicita de video_id (ignora a amostragem)")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="Todos os pendentes, alternando canais")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--media-resolution", default="MEDIA_RESOLUTION_LOW",
                        choices=["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"])
    parser.add_argument("--rpm", type=float, default=8.0, help="Requisicoes por minuto (respeite a cota do seu plano)")
    parser.add_argument("--max-video-hours", type=float, default=7.5,
                        help="Teto de horas de video nas ultimas 24 h, somando todos os runs (0 = sem teto, plano pago)")
    parser.add_argument("--retry-unavailable", action="store_true", help="Tentar de novo videos marcados not_available")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--consolidate", action="store_true", help="So regrava a Silver consolidada")
    args = parser.parse_args()

    if args.list_models:
        list_models(load_api_key())
        return
    source = resolve_source(args.source_csv)
    if args.consolidate:
        consolidate(source)
        return

    rows = list({r["video_id"]: r for r in csv.DictReader(open(source, encoding="utf-8"))}.values())  # sem duplicados
    done, unavailable = resolved_ids(args.retry_unavailable)
    pending = [r for r in rows if r["video_id"] not in done and r["video_id"] not in unavailable]
    if args.video_ids:
        wanted = set(args.video_ids)
        batch = [r for r in pending if r["video_id"] in wanted]
    elif args.all:
        batch = interleave_by_channel(pending, args.seed)
    else:
        batch = stratified_sample(pending, args.limit, args.seed)

    used_24h = seconds_sent_last_24h()
    budget = float("inf") if args.max_video_hours <= 0 else max(0.0, args.max_video_hours * 3600 - used_24h)
    seconds = sum(float(r.get("duration_seconds") or 0) for r in batch)
    mix = defaultdict(int)
    for r in batch:
        mix[(r["sector"], r["channel_type"])] += 1
    print(f"fonte: {source}")
    print(f"ja resolvidos: {len(done)} | indisponiveis pulados: {len(unavailable)} | pendentes: {len(pending)} | lote: {len(batch)}")
    print("mistura:", {f"{a}/{b}": n for (a, b), n in sorted(mix.items())})
    print(f"video no lote: {seconds / 3600:.2f} h | enviado nas ultimas 24 h: {used_24h / 3600:.2f} h | "
          f"orcamento agora: {'sem teto' if budget == float('inf') else f'{budget / 3600:.2f} h'}")
    print(f"tokens de entrada estimados (~100/s em resolucao baixa): {seconds * 100:,.0f}")
    if args.dry_run or not batch:
        return
    if budget <= 0:
        print("Orcamento de 24 h esgotado; rode de novo mais tarde (ou --max-video-hours 0 no plano pago).")
        return

    key = load_api_key()
    with Lock():
        run_dir = BRONZE / args.run_id
        (run_dir / "responses").mkdir(parents=True, exist_ok=False)
        prompt_hash = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()
        write_atomic(run_dir / "run_config.json", json.dumps({
            "run_id": args.run_id, "model": args.model, "media_resolution": args.media_resolution,
            "extractor_version": EXTRACTOR_VERSION, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash,
            "prompt": PROMPT, "sector_hints": SECTOR_HINTS, "schema": SCHEMA, "source_csv": str(source),
            "planned_video_ids": [r["video_id"] for r in batch],
            "usage_note": "URL publica do YouTube enviada ao Gemini; video nao baixado.",
        }, indent=2, ensure_ascii=False))

        interval = 60.0 / args.rpm
        used, consecutive_failures, stop_reason = 0.0, 0, "lote concluido"
        counts = defaultdict(int)
        manifest = open(run_dir / "manifest.jsonl", "a", encoding="utf-8")
        try:
            for n, row in enumerate(batch, 1):
                duration = float(row.get("duration_seconds") or 0)
                if used + duration > budget:
                    stop_reason = f"orcamento de {args.max_video_hours} h/24 h atingido"
                    break
                started = time.time()
                try:
                    status, payload, meta = call_gemini(key, args.model, row["video_id"], row["sector"], args.media_resolution)
                except DailyQuotaExceeded as error:
                    stop_reason = "cota diaria da API esgotada (429)"
                    print(f"  {row['video_id']}: {error}"[:300])
                    break
                record = {"video_id": row["video_id"], "channel_id": row["channel_id"], "sector": row["sector"],
                          "run_id": args.run_id, "model": args.model, "prompt_version": PROMPT_VERSION,
                          "collected_at": datetime.now(timezone.utc).isoformat(),
                          "latency_s": round(time.time() - started, 1), "duration_seconds": duration, **meta}
                if status == "ok":
                    # resposta salva antes do manifest: se cair aqui, a resposta valida ja impede nova chamada
                    write_atomic(run_dir / "responses" / f"{row['video_id']}.json", json.dumps(payload, ensure_ascii=False))
                    try:
                        problems = validate(parse_response(payload))
                        record["status"] = "invalid_json" if problems else "ok"
                        if problems:
                            record["error"] = "; ".join(problems)
                    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                        record["status"], record["error"] = "invalid_json", repr(error)
                        finish = (payload.get("candidates") or [{}])[0].get("finishReason")
                        if finish:
                            record["finish_reason"] = finish
                    record["usage"] = payload.get("usageMetadata", {})
                else:
                    record["status"] = status
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                os.fsync(manifest.fileno())
                counts[record["status"]] += 1
                used += duration
                consecutive_failures = 0 if record["status"] in ("ok", "not_available") else consecutive_failures + 1
                print(f"[{n}/{len(batch)}] {row['video_id']} {record['status']} ({record['latency_s']}s) "
                      f"| video enviado neste run: {used / 3600:.2f} h", flush=True)
                if consecutive_failures >= 5:
                    stop_reason = "circuit breaker: 5 falhas seguidas (veja o manifest)"
                    break
                time.sleep(max(0.0, interval - (time.time() - started)))
        except KeyboardInterrupt:
            stop_reason = "interrompido pelo usuario (Ctrl+C); tudo ate o ultimo video foi salvo"
        finally:
            manifest.close()

    tokens = defaultdict(int)
    for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        for k, v in json.loads(line).get("usage", {}).items():
            if isinstance(v, int):
                tokens[k] += v
    print(f"\nparada: {stop_reason}")
    print("status deste run:", dict(counts))
    print("tokens deste run:", dict(tokens))
    print("bronze:", run_dir)
    consolidate(source)
    remaining = len(pending) - counts.get("ok", 0)
    if remaining > 0:
        print(f"faltam ~{remaining} videos: rode o mesmo comando de novo para retomar.")


if __name__ == "__main__":
    main()
