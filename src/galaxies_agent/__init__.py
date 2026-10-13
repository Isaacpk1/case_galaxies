"""Agente preditivo de performance de YouTube Shorts (Inteli Academy x Galaxies).

Uso como funcao Python:

    from galaxies_agent import Predictor, CreativeInput
    predictor = Predictor()
    out = predictor.predict(CreativeInput(creative="https://www.youtube.com/shorts/<id>", caption="...",
                                          planned_datetime="2026-10-20T18:00", channel_id="UC...", sector="moda"))

Uso como tool MCP: `python -m galaxies_agent.mcp_server` (ver README).
"""

from .predictor import Predictor
from .schemas import ChannelStats, CreativeInput, PredictionOutput, RankingOutput

__all__ = ["Predictor", "CreativeInput", "ChannelStats", "PredictionOutput", "RankingOutput"]
