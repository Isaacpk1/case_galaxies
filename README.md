# Agente Preditivo de Performance em Midias Sociais

MVP para prever a performance de um YouTube Short antes da publicacao. O modelo recebe o criativo e o contexto do canal, estima curtidas ou visualizacoes, apresenta intervalo de confianca, ranking e os fatores que influenciaram a previsao.

O foco inicial e Brasil, com canais de moda e automotivo. A metrica principal do piloto e `like_count`; a decisao de produto e comparar criativos para o mesmo canal antes de investir midia.

## Arquitetura em uma frase

Coletamos metadados e metricas pela YouTube Data API, enriquecemos o mesmo `video_id` com thumbnail e transcricao, preservamos cada coleta por `run_id` e treinamos modelos tabulares em snapshots temporais sem vazamento.

```text
YouTube Data API ──┬── metadados + metricas ──────┐
                   ├── thumbnail_url ──► visual ──┼──► Feature Store ─► Gold ─► Modelo
video_id ──────────└── youtube-transcript-api ────┘
                                      transcricao
```

Consulte a [arquitetura ELT do MVP](docs/09-arquitetura-elt-mvp.md) para os contratos, regras de governanca e decisoes. O [roadmap multimodal](docs/08-roadmap-de-dados-multimodais.md) descreve as proximas etapas.

## Inicio rapido

Prerequisito: Python 3.11 ou superior e uma chave da YouTube Data API v3.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Preencha `YOUTUBE_API_KEY` no arquivo `.env`. A chave nunca deve ser versionada.

### 1. Coletar metadados e metricas

```powershell
.\.venv\Scripts\python.exe scripts\collect_data_discovery.py --sectors automotivo
```

Isso cria dados Bronze em `data/bronze/youtube/<run_id>/` e uma camada Silver de Shorts elegiveis em `data/silver/data_discovery/<run_id>/`.

### 2. Coletar transcricoes de um lote pequeno

O coletor salva apenas legendas e **nao baixa video ou audio**. Cada tentativa gera um manifesto com `downloaded`, `rate_limited`, `not_available`, `blocked` ou `failed`, pula videos ja resolvidos em runs anteriores e grava os pendentes em `pending_video_ids`.

| Provider | Origem da requisicao | Quando usar |
| --- | --- | --- |
| `transcript-api` (padrao) | seu IP | testes pontuais; bloqueado em IPs de cloud e apos volume alto |
| `yt-dlp` | seu IP, com PO Token (bgutil) | rede residencial "limpa"; exige Deno e o servidor bgutil aberto |
| `apify` | infraestrutura do fornecedor | quando o IP local esta bloqueado; exige `APIFY_TOKEN` e consome creditos |

Teste do Apify com 5 videos (custo estimado impresso antes de rodar; `--max-cost-usd` padrao US$ 2):

```powershell
.\.venv\Scripts\python.exe scripts\collect_youtube_captions_pilot.py --source-run-id 20260929T210254Z --provider apify --limit 5
```

Coleta local, com pausas e circuit breaker:

```powershell
.\.venv\Scripts\python.exe scripts\collect_youtube_captions_pilot.py --source-run-id 20260929T210254Z --provider yt-dlp --limit 10
```

Para validar varios runs de transcricao de uma vez na Silver, passe todos em `materialize_silver.py --transcript-run-id RUN_1 RUN_2 ...`; o melhor resultado por `video_id` e mantido.

### 3. Materializar thumbnails para o mesmo lote

```powershell
.\.venv\Scripts\python.exe scripts\collect_youtube_thumbnails.py --source-run-id 20260929T210254Z --sector automotivo --limit 10
```

O coletor baixa a thumbnail referenciada pela API, valida a imagem e gera hash, dimensoes e status em `data/silver/thumbnail_manifest/<run_id>/thumbnail_manifest.json`.

### 4. Materializar e avaliar o baseline

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pilot_dataset.py
.\.venv\Scripts\python.exe scripts\run_baseline_experiments.py
```

### 5. Tool preditiva (funcao Python + servidor MCP)

O modelo servido e o CB_EMB do E-08 (CatBoost no desvio sobre a mediana do canal + embedding SigLIP 2 da thumbnail), treinado so com o split de treino. Roda em Linux, macOS ou Colab; no Windows do Inteli as DLLs nativas (CatBoost, Pydantic) sao bloqueadas.

```bash
pip install -r requirements-tool.txt
python scripts/train_tool_model.py --validate      # gera artifacts/tool_model/<versao>/ e LATEST.json
PYTHONPATH=src python -m pytest -q tests/test_tool.py
PYTHONPATH=src python -m galaxies_agent.mcp_server  # servidor MCP (stdio)
```

Como funcao Python:

```python
from galaxies_agent import Predictor, CreativeInput
out = Predictor().predict(CreativeInput(
    creative="https://www.youtube.com/shorts/<id>",   # ou caminho/URL da thumbnail
    caption="Titulo planejado", planned_datetime="2026-10-20T18:00",
    channel_id="UC...", sector="automotivo"))
print(out.predicted_metric, out.interval_low, out.interval_high, out.confidence_flag, out.explanation)
```

Configuracao de cliente MCP (ex.: Claude Desktop), com `cwd` na raiz do repo:

```json
{"mcpServers": {"galaxies-preditivo": {"command": "python", "args": ["-m", "galaxies_agent.mcp_server"],
  "env": {"PYTHONPATH": "src"}}}}
```

| Tool | Entrada | Saida |
| --- | --- | --- |
| `prever_criativo` | `CreativeInput` (criativo, legenda, data planejada, `channel_id` ou `channel_stats`, setor) | previsao de curtidas, intervalo de 80%, `confidence_flag` e motivos, desempenho relativo ao canal e faixa, `top_drivers` (SHAP), Shorts comparaveis, explicacao, features usadas |
| `ranquear_criativos` | 2 a 5 `CreativeInput` | mesma saida, ordenada pelo desempenho relativo ao canal |
| `historico_canal` | `channel_id` | resumo e ultimos Shorts do canal |
| `info_modelo` | - | versao, dados de treino, alvo e metodo do intervalo |

Regras da tool: o numero vem sempre do modelo; o B0 usa so Shorts do canal publicados antes da data planejada (testado contra o Gold); intervalo por calibracao conformal (cobertura de 76% fora da amostra para meta de 80%); sem embedding da thumbnail (sem `torch`/`transformers` ou sem acesso ao Hugging Face e fora do cache), usa o modelo reserva e marca confianca baixa. Para criativos novos fora do cache, calcule o embedding no Colab ou passe um `.npz` extra em `Predictor(extra_embedding_cache=[...])`.

### 6. Agente (LangGraph)

`src/galaxies_agent/agent.py` orquestra as tools: interpretar o pedido -> historico do canal -> atributos do video (rubrica Gemini, quando coletada) -> prever/ranquear -> explicar -> verificar. O LLM (Gemini via `langchain-google-genai`, chave `GEMINI_API_KEY`) so interpreta o pedido e escreve a explicacao; o no `verificar` confere que todo numero citado existe na saida das tools e, se nao existir, troca o texto pela explicacao deterministica do SHAP. Sem chave, o agente roda em modo deterministico com entrada estruturada.

```bash
PYTHONPATH=src python -m galaxies_agent.agent "Qual destes Shorts priorizar?" \
  --criativos '[{"creative": "https://www.youtube.com/shorts/<id>", "planned_datetime": "2026-10-20T18:00", "channel_id": "UC...", "sector": "moda"}]'
```

## Estrutura de dados

| Camada | Conteudo | Regra principal |
| --- | --- | --- |
| Bronze | Respostas brutas, transcricoes e manifestos por `run_id` | Nunca sobrescrever coleta existente |
| Silver | Shorts elegiveis, status de transcricao e metadados validados | Falha de uma modalidade nao exclui o Short |
| Feature Store | Atributos de metadados, texto, thumbnail e fala | Versao de extrator e `source_hash` obrigatorios |
| Gold | Snapshot temporal de treino | Alvo separado de features e sem vazamento |

## Regras inegociaveis

1. `like_count`, `view_count` e `comment_count` do proprio video sao alvo, nunca feature pre-publicacao.
2. Estatisticas do canal usam somente videos anteriores ao criativo previsto.
3. Toda coleta e rastreada por `run_id`, `video_id`, data e hash quando houver arquivo.
4. `youtube-transcript-api` nao tem cobertura garantida; seu status deve virar flag de feature, nao motivo para apagar uma linha.
5. A YouTube Data API e usada para metadados e metricas; ela nao e um coletor de video.

## Experimentos obrigatorios

A escolha do modelo nao sera feita por split aleatorio. Cada experimento deve usar validacao temporal rolling-origin e, quando possivel, teste em canais nao vistos.

1. Metadados e historico do canal;
2. Metadados + texto;
3. Metadados + texto + thumbnail;
4. Metadados + texto + thumbnail + transcricao;
5. Comparacao contra a mediana historica do canal.

Metricas: MAE em escala log, MAPE, Spearman, acuracia par-a-par e cobertura do intervalo de predicao.

## Limites atuais

- A transcricao vem de uma rota nao oficial e pode falhar por indisponibilidade ou bloqueio de IP (ver D-011); o provider `apify` depende de um fornecedor terceiro.
- A thumbnail e uma modalidade visual estatica; frames e audio completo nao pertencem a esta primeira arquitetura.
- Shorts sao inferidos por duracao de 1 a 180 segundos, pois a API publica nao expoe uma flag exclusiva.
- As metricas sao uma fotografia de coleta; o alvo por horizonte fixo e uma evolucao futura.

## Documentacao

- [Entendimento do problema](docs/01-entendimento-do-problema.md)
- [Entendimento dos dados](docs/03-entendimento-dos-dados.md)
- [Hipoteses de modelagem](docs/05-hipoteses-de-modelagem.md)
- [Metricas e diagnosticos](docs/06-metricas-e-diagnosticos.md)
- [Registro de features](docs/07-registro-de-features.md)
- [Roadmap multimodal](docs/08-roadmap-de-dados-multimodais.md)
- [Arquitetura ELT do MVP](docs/09-arquitetura-elt-mvp.md)
- [Stack e selecao de modelo](docs/10-stack-e-selecao-de-modelo.md)
