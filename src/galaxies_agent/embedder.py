"""Embedding SigLIP 2 da thumbnail, com cache e o mesmo pre-processamento do treino.

Ordem de busca para um criativo:
1. cache por video_id (embeddings ja calculados na Silver ou em um .npz extra, ex.: criativos de demo
   calculados no Colab);
2. calculo ao vivo com SigLIP 2 (exige torch + transformers e acesso ao Hugging Face);
3. sem embedding -> o preditor usa o modelo reserva e marca confianca baixa.

Pre-processamento (igual ao pacote do E-08): se a thumbnail 16:9 for um frame vertical com laterais
desfocadas (pillarbox), recorta o painel central 9:16; depois redimensiona o lado menor para 256 px.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|/vi/)([A-Za-z0-9_-]{11})")
PILLAR_STRONG_RATIO = 0.50
PILLAR_WEAK_RATIO = 0.65
PILLAR_SEAM = 0.05


def youtube_id(creative: Optional[str]) -> Optional[str]:
    if not creative:
        return None
    match = YOUTUBE_ID.search(creative)
    if match:
        return match.group(1)
    return creative if re.fullmatch(r"[A-Za-z0-9_-]{11}", creative) and not Path(creative).exists() else None


def is_pillarbox(luma: np.ndarray) -> bool:
    """Mesma regra de scripts/extract_thumbnail_features.py (pillarbox_box)."""
    h, w = luma.shape
    pw = h * 9 / 16
    if w < pw * 1.3:
        return False
    x0, x1 = int(round((w - pw) / 2)), int(round((w + pw) / 2))
    inner = luma[:, x0 + 5 : x1 - 5]
    outer = np.concatenate([luma[:, : max(x0 - 5, 1)], luma[:, x1 + 5 :]], axis=1)
    if inner.size == 0 or outer.size == 0:
        return False
    ratio = float(outer.mean()) / (float(inner.mean()) + 1e-3)
    seam = 0.0
    for b in (x0, x1):
        for d in range(-4, 5):
            c = b + d
            if 2 <= c < w - 2:
                seam = max(seam, float(np.median(np.abs(luma[:, c + 1] - luma[:, c - 2]))))
    return ratio < PILLAR_STRONG_RATIO or (ratio < PILLAR_WEAK_RATIO and seam >= PILLAR_SEAM)


def preprocess(image):
    from PIL import Image

    image = image.convert("RGB")
    luma = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    w, h = image.size
    if is_pillarbox(luma):
        pw = round(h * 9 / 16)
        x = (w - pw) // 2
        image = image.crop((x, 0, x + pw, h))
    scale = 256 / min(image.size)
    return image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.BICUBIC)


def load_image(creative: str):
    from PIL import Image

    vid = youtube_id(creative)
    urls = [creative] if creative.lower().startswith("http") and not vid else []
    if vid:
        urls = [f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg", f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return Image.open(io.BytesIO(resp.read()))
        except Exception:
            continue
    path = Path(creative)
    if path.exists():
        return Image.open(path)
    raise FileNotFoundError(f"nao consegui abrir a imagem do criativo: {creative}")


class ThumbnailEmbedder:
    def __init__(self, model_id: str, cache_files: list[Path]):
        self.model_id = model_id
        self.cache: dict[str, np.ndarray] = {}
        for path in cache_files:
            if path and Path(path).exists():
                data = np.load(path, allow_pickle=True)
                self.cache.update(zip(data["video_id"].astype(str), data["embeddings"].astype(np.float32)))
        self._model = None
        self.live_error: Optional[str] = None

    def _load_live(self):
        if self._model is None and self.live_error is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModel, AutoProcessor

                self._model = AutoModel.from_pretrained(self.model_id).eval()
                self._processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as error:  # sem torch, sem HF, sem rede
                self.live_error = f"{type(error).__name__}: {error}"[:300]
        return self._model

    def embed(self, creative: Optional[str]) -> tuple[Optional[np.ndarray], str]:
        """(vetor L2-normalizado ou None, origem)."""
        if not creative:
            return None, "sem_imagem"
        vid = youtube_id(creative)
        if vid and vid in self.cache:
            return self.cache[vid], "cache"
        model = self._load_live()
        if model is None:
            return None, f"indisponivel ({self.live_error})"
        import torch

        image = preprocess(load_image(creative))
        inputs = self._processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        vector = torch.nn.functional.normalize(out.float(), dim=-1)[0].numpy()
        if vid:
            self.cache[vid] = vector
        return vector, "siglip2_ao_vivo"
