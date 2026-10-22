"""Registra o servidor MCP do projeto no Claude Desktop (claude_desktop_config.json), so com a stdlib.

O Claude Desktop nao usa `cwd`: o comando, o PYTHONPATH e a raiz do repo precisam ser caminhos absolutos. Este script
descobre esses caminhos, testa se o Python escolhido carrega o modelo (catboost/pandas) e grava a entrada
"galaxies-preditivo" preservando os outros servidores do arquivo (copia de seguranca em .bak).

    python scripts/setup/configure_claude_desktop.py                  # usa o Python que rodou o script
    python scripts/setup/configure_claude_desktop.py --python .venv/Scripts/python.exe --gemini-key <chave>
    python scripts/setup/configure_claude_desktop.py --dry-run        # so mostra o JSON

Depois: feche o Claude Desktop por completo (inclusive da bandeja) e abra de novo.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "galaxies-preditivo"
CHECK = ("from galaxies_agent.predictor import Predictor; "
         "info = Predictor().model_info(); print(info.get('model_version') or info.get('versao') or 'ok')")


def config_path() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def check(python: Path, env: dict) -> str:
    """Roda o carregamento do modelo no Python escolhido; falha cedo se catboost/pandas nao carregarem."""
    proc = subprocess.run([str(python), "-c", CHECK], env={**os.environ, **env}, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise SystemExit(f"o Python {python} nao carregou o modelo:\n{tail}\n"
                         "instale as dependencias (pip install -r requirements-tool.txt) ou use --skip-check")
    return proc.stdout.strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="interpretador com as dependencias")
    parser.add_argument("--gemini-key", help="opcional: chave do Gemini para a rubrica ao vivo de Shorts novos")
    parser.add_argument("--config", type=Path, default=None, help="caminho do claude_desktop_config.json")
    parser.add_argument("--skip-check", action="store_true", help="nao testa o carregamento do modelo")
    parser.add_argument("--dry-run", action="store_true", help="so imprime a entrada, nao grava")
    args = parser.parse_args()

    python = args.python.resolve()
    if not python.exists():
        found = shutil.which(str(args.python))
        if not found:
            raise SystemExit(f"Python nao encontrado: {args.python}")
        python = Path(found).resolve()
    env = {"PYTHONPATH": str(ROOT / "src"), "GALAXIES_ROOT": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    if args.gemini_key:
        env["GEMINI_API_KEY"] = args.gemini_key
    entry = {"command": str(python), "args": ["-m", "galaxies_agent.mcp_server"], "env": env}

    if not args.skip_check:
        print(f"modelo carregado: {check(python, env)}")
    shown = {**entry, "env": {k: ("<definida>" if k == "GEMINI_API_KEY" else v) for k, v in env.items()}}
    print(json.dumps({"mcpServers": {NAME: shown}}, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    path = args.config or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists() and path.read_text(encoding="utf-8").strip():
        data = json.loads(path.read_text(encoding="utf-8"))
        shutil.copy2(path, path.with_suffix(".json.bak"))
    data.setdefault("mcpServers", {})[NAME] = entry
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado em {path}. Feche o Claude Desktop por completo e abra de novo.")


if __name__ == "__main__":
    main()
