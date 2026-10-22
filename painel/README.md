# Painel Galaxies Shorts (front)

Painel de resultados do agente preditivo: o que o modelo entrega, onde ele erra, como deve ser um Short e os bastidores
da validacao. Publicado como artifact do tipo Dashboard: <https://claude.ai/artifact/DeDmDXainSZacMnHzJehxZ>

```
index.html      pagina do painel (HTML + CSS + JS, d3 v7; sem build)
dados/*.json    um arquivo por dataset, no mesmo nome que o `data-source` do HTML
```

## Como roda

A pagina e feita para o runtime de Dashboard do artifact: ela nao busca os JSON por conta propria. O runtime entrega
cada dataset pelo `data-source` e a pagina usa `dash.data(id)`, `dash.calc('resumo', ...)` e `dash.onData(draw)`; as
cores saem das variaveis do tema (`--color-*`, `--cds-*`). Por isso **abrir o `index.html` direto no navegador mostra a
pagina sem numeros**: para ver o painel, use o link do artifact acima.

Para republicar ou recriar o painel: criar um artifact do tipo Dashboard, enviar `index.html` como `files/index.html` e
cada `dados/<id>.json` como o dataset `<id>`.

## De onde vem cada dataset

| Dataset | Fonte |
| --- | --- |
| `base` | numeros gerais do pacote servido (`artifacts/tool_model/LATEST.json` + `metadata.json`) e do Gold v2.1 |
| `modelos_teste` | `metadata.json` do pacote, bloco `final_test` (teste do E-15) |
| `evolucao` | E-13 (Gold v1) vs E-14 e E-15 (Gold v2.1), em `artifacts/feature_experiments/` e `docs/07` |
| `ordenacao` | acerto por pares do teste do E-15 e do zero-shot (`artifacts/feature_experiments/llm_zero_shot/`) |
| `recortes`, `canais`, `maiores_erros` | `artifacts/feature_experiments/e14_e15_gold_v2/e15/error_analysis.json` |
| `intervalo` | `artifacts/feature_experiments/interval_calibration/report.json` |
| `clusters` | `artifacts/feature_experiments/e16_descriptor_clusters/report.json` |
| `importancia`, `atributos` | `artifacts/feature_experiments/what_works/` (`scripts/experimentos/analyze_what_works.py`) |
| `e17` | `artifacts/feature_experiments/e17_video_frames/report.json` |

Os JSON aqui sao a copia publicada em 21/10/2026. Ao reavaliar o modelo, gere os relatorios de novo e atualize os
arquivos correspondentes antes de republicar.
