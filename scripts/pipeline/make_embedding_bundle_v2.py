"""Empacota as thumbnails do Gold v2 que ainda nao tem embedding SigLIP 2, para rodar no Colab.

Aplica o MESMO pre-processamento da v1 (galaxies_agent.embedder.preprocess: recorta o painel 9:16 quando a
thumbnail e pillarbox e redimensiona o lado menor para 256 px) e grava thumbs/<video_id>.jpg + ids.csv.
O notebook 04_modelo_colab/12_embedding_thumbnails_v2.ipynb gera embeddings.npz no formato da v1
(video_id, embeddings L2-normalizados, source), que vai para data/silver/features/thumbnail_siglip2_v2/<run>/.

    python scripts/pipeline/make_embedding_bundle_v2.py --gold gold_v2_<...> --thumbnail-runs <RUN_AUTO> <RUN_MODA> --part 1 --parts 5
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from galaxies_agent.embedder import preprocess  # noqa: E402

V1_EMB = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--thumbnail-runs", nargs="+", required=True)
    parser.add_argument("--part", type=int, default=1, help="parte a gerar (1..parts)")
    parser.add_argument("--parts", type=int, default=1, help="dividir em N arquivos (o Colab le todos)")
    args = parser.parse_args()

    gold = pd.read_csv(ROOT / "data" / "gold" / "snapshots" / args.gold / "training_snapshot.csv", usecols=["video_id"])
    done = set(np.load(V1_EMB, allow_pickle=True)["video_id"].astype(str))
    files: dict[str, Path] = {}
    for run in args.thumbnail_runs:
        for path in (ROOT / "data" / "bronze" / "thumbnails" / "youtube" / run / "files").glob("*"):
            files.setdefault(path.stem, path)
    todo = [v for v in gold["video_id"].astype(str) if v not in done]
    missing = [v for v in todo if v not in files]
    print(f"Gold: {len(gold)} | ja com embedding (v1): {len(gold) - len(todo)} | a calcular: {len(todo)} | sem thumbnail: {len(missing)}")

    todo = [v for v in todo if v in files][args.part - 1::args.parts]
    out = ROOT / ".tmp_transfer" / f"emb_v2_bundle_part{args.part}of{args.parts}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    with tarfile.open(out, "w:gz") as tar:
        for i, vid in enumerate(todo):
            try:
                image = preprocess(Image.open(files[vid]))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=92)
                info = tarfile.TarInfo(f"emb_v2/thumbs/{vid}.jpg")
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(info, buffer)
                rows.append(vid)
            except Exception as error:  # imagem corrompida
                failed.append((vid, str(error)[:120]))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} imagens", flush=True)
        ids = io.StringIO()
        writer = csv.writer(ids)
        writer.writerow(["video_id"])
        writer.writerows([[v] for v in rows])
        data = ids.getvalue().encode()
        info = tarfile.TarInfo(f"emb_v2/ids_part{args.part}.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(f"{len(rows)} imagens em {out} ({out.stat().st_size / 1e6:.0f} MB); falhas: {len(failed)}; sem thumbnail: {len(missing)}")
    if missing or failed:
        Path(str(out) + ".faltando.csv").write_text("video_id,motivo\n" + "".join(f"{v},sem_thumbnail\n" for v in missing)
                                                    + "".join(f"{v},{e}\n" for v, e in failed), encoding="utf-8")


if __name__ == "__main__":
    main()
