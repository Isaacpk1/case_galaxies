# Tabela de consumo do piloto

## Objetivo

`data/gold/pilot_consumption_table.csv` e a tabela tabular para os primeiros experimentos do modelo. Cada linha representa um Short elegivel de um dos 12 canais do piloto que ja possui pelo menos cinco posts anteriores do mesmo canal.

Ela e gerada de forma reproduzivel pelo comando abaixo:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\prepare_pilot_dataset.py
```

## Regras do contrato

- Cada linha e identificada por `video_id` e preserva `channel_id`, `channel_name` e `published_at` para rastreabilidade e validacao temporal.
- As colunas de contexto e de conteudo resumido sao disponiveis antes da publicacao.
- `channel_prior_*`, `days_since_previous_post` e `baseline_log_likes` usam somente posts anteriores ao video da linha.
- As colunas iniciadas por `target_` e `like_count` sao resultados. Podem ser usadas apenas para treino e avaliacao, nunca como entrada de uma previsao de video nao publicado.
- `split` reserva os 20% mais recentes de cada canal para teste. O restante forma o treino. Assim, nenhum post futuro do mesmo canal aparece no treino ao avaliar um post do teste.

## Colunas

| Grupo | Colunas |
| --- | --- |
| Rastreabilidade | `video_id`, `channel_id`, `channel_name`, `published_at`, `split` |
| Contexto | `sector`, `market`, `channel_type` |
| Formato e texto resumido | `duration_seconds`, `publish_hour_utc`, `publish_weekday`, `title_length`, `description_length`, `tag_count` |
| Historico pre-publicacao | `channel_prior_posts`, `channel_prior_median_likes`, `days_since_previous_post`, `baseline_log_likes` |
| Alvos | `target_log_likes`, `target_relative_to_prior_median` |

O texto bruto, descricao, tags e URL de thumbnail permanecem em `pilot_dataset.csv`. Eles nao entram na primeira tabela tabular para evitar misturar a etapa de atributos multimodais com o baseline inicial.
