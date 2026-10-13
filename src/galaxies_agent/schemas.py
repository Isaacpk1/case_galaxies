"""Contrato da tool (secao 7 do TAP), validado com Pydantic.

Entrada minima: criativo, legenda planejada, data/hora planejada, canal (id ou estatisticas) e setor.
Saida: previsao, intervalo, confianca, desempenho relativo ao canal, drivers, explicacao e comparaveis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Sector = Literal["moda", "beleza", "automotivo"]


class ChannelStats(BaseModel):
    """Historico informado pelo cliente quando o canal nao esta na base (agnostico a base)."""

    channel_name: Optional[str] = None
    channel_type: Literal["brand", "creator"] = "brand"
    prior_likes: list[int] = Field(..., min_length=1, description="Curtidas dos Shorts anteriores do canal")
    last_post_at: Optional[datetime] = Field(None, description="Data do ultimo Short publicado")
    median_duration_seconds: Optional[float] = None


class CreativeInput(BaseModel):
    creative: Optional[str] = Field(
        None, description="Caminho ou URL da thumbnail/imagem, URL ou id de video do YouTube. Vazio = so texto.")
    creative_id: Optional[str] = Field(None, description="Rotulo livre para identificar o criativo na resposta")
    caption: str = Field("", description="Titulo/legenda planejada")
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    planned_datetime: datetime = Field(..., description="Data e hora previstas de publicacao (sem fuso = America/Sao_Paulo)")
    channel_id: Optional[str] = None
    channel_stats: Optional[ChannelStats] = None
    sector: Sector
    duration_seconds: Optional[float] = Field(None, gt=0, le=180, description="Duracao do Short; vazio = mediana do canal")

    @model_validator(mode="after")
    def need_channel(self) -> "CreativeInput":
        if not self.channel_id and not self.channel_stats:
            raise ValueError("informe channel_id (canal da base) ou channel_stats (historico do canal)")
        return self


class Driver(BaseModel):
    feature: str
    label: str
    value: str
    effect_log: float = Field(..., description="Contribuicao SHAP no desvio previsto (log curtidas)")
    direction: Literal["aumenta", "reduz"]


class Comparable(BaseModel):
    video_id: str
    title: str
    channel_name: str
    published_at: str
    likes: int
    relative_to_channel: float
    similarity: float
    url: str


class PredictionOutput(BaseModel):
    creative_id: Optional[str]
    metric: Literal["like_count"] = "like_count"
    predicted_metric: int
    interval_low: int
    interval_high: int
    interval_level: float
    confidence_flag: Literal["alta", "baixa"]
    confidence_reasons: list[str]
    channel_baseline_likes: int
    relative_to_channel: float = Field(..., description="Previsao / mediana historica do canal")
    performance_band: Literal["abaixo", "na_media", "acima", "viral"]
    top_drivers: list[Driver]
    explanation: str
    comparables: list[Comparable]
    features_used: dict
    model_version: str
    embedding_source: str


class RankedCreative(PredictionOutput):
    rank: int


class RankingOutput(BaseModel):
    ranking: list[RankedCreative]
    note: str
