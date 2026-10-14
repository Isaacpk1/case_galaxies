# Agente Preditivo de Performance em Midias Sociais

Inteli Academy x Galaxies (TAPI). Agente que preve, **antes da publicacao**, as curtidas de um YouTube Short de moda ou automotivo, com intervalo de predicao, ranking entre criativos e explicacao dos fatores. O numero vem de um modelo treinado em dados observados; o LLM so orquestra e explica. O fluxo e exposto como funcao Python, servidor MCP e agente LangGraph.

**Resultado principal (teste final, aberto uma unica vez, 612 Shorts):** o modelo final (CatBoost sobre o desvio da mediana do canal, com embedding SigLIP 2 da thumbnail, Shorts comparaveis e rubrica de video + fala do Gemini) erra 13% menos que a mediana historica do canal (MAE 0,746 vs 0,858 em log), acerta a ordem de 58% dos pares de Shorts do mesmo canal (mediana do canal: 44%) e seu intervalo de 80% cobre 79,9% dos casos. Na validacao temporal, a rubrica Gemini + comparaveis melhoraram o modelo com p = 0,003 (21 de 26 canais). Em canais nunca vistos, o modelo ainda ordena acima do acaso (55%, p = 0,002).

---

## 1. Problema e recorte

| Item | Decisao |
| --- | --- |
| Pergunta | Dado um criativo ainda nao publicado, quantas curtidas ele deve ter **neste canal**? Qual de varios criativos priorizar? |
| Plataforma | YouTube Shorts (YouTube Data API v3) |
| Setores | Moda e automotivo (o TAP pede ao menos dois de moda, beleza e automotivo) |
| Metrica | `like_count` |
| Unidade | Criativo em um canal especifico |
| Baseline (B0) | Mediana historica de curtidas do canal (so posts anteriores) |
| Uso | Tool consumida pela persona sintetica da Galaxies (MCP) |

Detalhes e decisoes em [docs/01](docs/01-entendimento-do-problema.md).

## 2. Base de dados

| Item | Valor |
| --- | --- |
| Origem | YouTube Data API v3 (metadados e metricas publicas); thumbnails publicas; rubrica de video e transcricao via Gemini a partir da URL publica |
| Periodo | 01/04/2025 a 30/08/2026 (18 meses de historico, Shorts com pelo menos 30 dias na data da coleta, 28-29/09/2026) |
| Volume | 3.004 Shorts, 26 canais (moda 1.589, automotivo 1.415; marcas e criadores) - Gold `gold_v1_20261002T013000Z_663f502d` |
| Split | Treino 2.392 / teste 612 (os 20% mais recentes de cada canal); **teste aberto uma unica vez**, depois da decisao do E-13 |
| Licenca e uso | Dados publicos coletados sob os Termos da YouTube API; nao redistribuidos (camadas `data/` fora do git). Gemini recebe apenas URLs publicas; nenhum video e baixado |

**O que a base nao representa:** beleza (fora do recorte); canais com menos de 30 Shorts; metricas de distribuicao (impressoes, CTR, retencao - nao existem na API publica); curtidas como "fotografia" da data de coleta, nao em horizonte fixo; mercado brasileiro em portugues.

Mais em [docs/03](docs/03-entendimento-dos-dados.md) e [docs/09](docs/09-arquitetura-elt-mvp.md).

## 3. Arquitetura

```text
YouTube Data API ──► Bronze (JSON por run_id) ──► Silver (Shorts elegiveis, manifestos)
thumbnail_url ─────► Bronze thumbnails ─────────► SigLIP 2 (embedding) + features visuais/OCR
URL publica ───────► Gemini (rubrica + fala) ───► Silver gemini_video
                                   │
                                   ▼
              Feature Store versionada (hash, available_before_publish)
                                   │
                                   ▼
              Gold: snapshot temporal (alvo separado, checagens antivazamento)
                                   │
                                   ▼
     CatBoost final (desvio sobre B0): PCA do embedding + comparaveis visuais + rubrica Gemini
     + intervalo conformal + SHAP  (reservas: sem rubrica / sem imagem)
                                   │
            ┌──────────────────────┼───────────────────────┐
            ▼                      ▼                       ▼
   Funcao Python (Predictor)   Servidor MCP          Agente LangGraph
   contrato Pydantic           4 tools               interpretar → atributos do video (Gemini ao vivo)
                                                     → prever → explicar → verificar numeros
```

Camadas e contratos: [docs/09](docs/09-arquitetura-elt-mvp.md). Decisao de stack: [docs/10 (ADR-001)](docs/10-stack-e-selecao-de-modelo.md).

## 4. Instalacao e execucao

Python 3.11+. No Windows do Inteli a politica de aplicativos bloqueia DLLs nativas (pandas, CatBoost, Pydantic): **os coletores rodam no Windows** (so biblioteca padrao); **modelo, tool e agente rodam em Linux, macOS ou Colab**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # coleta e pipeline
pip install -r requirements-tool.txt     # modelo, tool, agente (Linux/Colab)
Copy-Item .env.example .env              # YOUTUBE_API_KEY e GEMINI_API_KEY (nunca versionar)
```

| Etapa | Comando |
| --- | --- |
| 1. Metadados e metricas | `python scripts/coleta/collect_data_discovery.py --sectors automotivo` |
| 2. Silver (varios runs) | `python scripts/pipeline/materialize_silver.py --youtube-run-id <RUN_AUTO> <RUN_MODA> --thumbnail-run-id ... --silver-run-id <ID>` |
| 3. Thumbnails | `python scripts/coleta/collect_youtube_thumbnails_stdlib.py --source-run-id <RUN> --sector moda` |
| 4. Rubrica de video + fala (Gemini) | `python scripts/coleta/collect_gemini_video_features.py --all` (retomavel, sem duplicados; `--dry-run` mostra o status) |
| 5. Feature Store e Gold | `python scripts/pipeline/build_feature_store.py --silver-run-id <ID>` e `python scripts/pipeline/build_gold_snapshot.py --silver-run-id <ID>` |
| 6. Embedding da thumbnail | `notebooks/04_modelo_colab/08_e08_catboost_residual_multimodal.ipynb` (Colab, GPU) |
| 7. Experimentos | notebooks 05-10 e `scripts/experimentos/` (ver secao 7); E-13: `python scripts/experimentos/run_e13_final.py` e `--report` |
| 7b. Teste final (uma vez) e erros | `python scripts/experimentos/run_e13_test_final.py` (recusa rodar de novo) e `python scripts/experimentos/analyze_e13_errors.py` |
| 8. Pacote do modelo da tool | `python scripts/modelo/train_tool_model.py` (todo o Gold; `--train-only` reproduz o modelo do teste) |
| 9. Testes | `PYTHONPATH=src python -m pytest -q tests/` |
| 10. Servidor MCP | `PYTHONPATH=src python -m galaxies_agent.mcp_server` |
| 11. Agente | `PYTHONPATH=src python -m galaxies_agent.agent "Qual destes Shorts priorizar?" --criativos '[...]'` |
| 12. Demo | `notebooks/05_demo/11_demo_agente.ipynb` (Colab; arraste `demo_bundle.tar.gz` e cadastre `GEMINI_API_KEY` nos Secrets) |

A coleta de legendas antiga (transcript-api/yt-dlp/apify) foi substituida pelo Gemini por bloqueio de IP (D-011); os scripts sem uso foram removidos em 14/10/2026 e continuam no historico do git.

### Uso como funcao Python

```python
from galaxies_agent import Predictor, CreativeInput
out = Predictor().predict(CreativeInput(
    creative="https://www.youtube.com/shorts/<id>",     # ou caminho/URL da thumbnail
    caption="Titulo planejado", planned_datetime="2026-10-20T18:00",
    channel_id="UC...", sector="automotivo"))           # canal fora da base: channel_stats={...}
print(out.predicted_metric, out.interval_low, out.interval_high, out.confidence_flag, out.explanation)
print(out.model_variant, out.rubric_source, out.video_attributes)   # o que o Gemini viu no video
```

### Uso via MCP (ex.: Claude Desktop; `cwd` na raiz do repo)

```json
{"mcpServers": {"galaxies-preditivo": {"command": "python", "args": ["-m", "galaxies_agent.mcp_server"],
  "env": {"PYTHONPATH": "src"}}}}
```

| Tool | Entrada | Saida |
| --- | --- | --- |
| `prever_criativo` | criativo, legenda, data planejada, `channel_id` ou `channel_stats`, setor | `predicted_metric`, `interval_low/high` (80%), `confidence_flag` + motivos, `relative_to_channel`, faixa, `top_drivers` (SHAP), `comparables`, `explanation`, `video_attributes` (rubrica do Gemini), `model_variant`, features usadas |
| `ranquear_criativos` | 2 a 5 criativos | mesma saida, ordenada pelo desempenho relativo ao canal |
| `historico_canal` | `channel_id` | resumo e ultimos Shorts do canal |
| `info_modelo` | - | versao, dados de treino, resultado do teste final, modelos e features, alvo, metodo do intervalo |

### Agente (LangGraph)

`interpretar` (LLM transforma o pedido em `CreativeInput`) → `contextualizar` (historico do canal) → `atributos` (o Gemini assiste ao Short pela URL publica e preenche a rubrica do TAP + transcricao; Shorts da base usam o cache) → `prever`/`ranquear` (modelo) → `explicar` (LLM) → `verificar` (todo numero do texto precisa existir na saida das tools; senao, usa a explicacao deterministica do SHAP). Sem chave do Gemini, a previsao usa o modelo reserva sem rubrica (confianca baixa) e a explicacao e a deterministica.

## 5. Modelo e configuracao

| Item | Configuracao |
| --- | --- |
| Algoritmo | CatBoost, perda MAE (modelo final = CB_FINAL do E-13) |
| Alvo | `log(1 + curtidas) - log(1 + mediana das curtidas dos Shorts anteriores do canal)` |
| Features | historico do canal (B0, posts anteriores, dias desde o anterior), contexto (duracao, setor, marca/criador, hora, dia da semana, canal), texto resumido (tamanhos, tags), **embedding SigLIP 2 da thumbnail (PCA 16)**, **comparaveis** (desvio medio dos 5 Shorts anteriores do canal com thumbnail mais parecida, similaridade, novidade visual) e **rubrica Gemini** (8 notas 0-10, 8 sinais sim/nao, tom de narracao, quantidade e ritmo de fala) |
| Hiperparametros | depth 6, learning_rate 0,03, l2_leaf_reg 5, random_strength 2, bagging_temperature 2, 1.390 arvores (busca aleatoria em holdout cronologico, E-08) |
| Reservas | sem rubrica (CB_EMB: criativo que nao e Short do YouTube, sem chave ou Gemini fora) e sem imagem (CB_BASE); a tool marca confianca baixa e diz o motivo |
| Rubrica do video | `gemini-3.5-flash-lite`, URL publica do YouTube (o video nao e baixado), resolucao baixa, JSON com schema fixo (`rubrica_tap_v1`), temperatura 0; ~5,5 mil tokens por Short |
| Treino do pacote | todo o Gold (3.004 Shorts, 26 canais), depois do teste final; `--train-only` reproduz o modelo avaliado |
| Intervalo | 80%, conformal: quantis 10%/90% dos residuos fora da amostra do rolling-origin (E-13), por tipo de canal; cobriu 79,9% no teste final |
| Explicacao | SHAP por previsao (embedding agrupado; rubrica e comparaveis com rotulos legiveis) + Shorts comparaveis por similaridade visual |
| Pacote | `artifacts/tool_model/<versao>/` (modelos, PCA, calibracao, historico, comparaveis, `metadata.json` com hashes) |

## 6. Metodologia

- **Antivazamento:** curtidas, views e comentarios do proprio video nunca sao features; historico so com posts anteriores; checado no build do Gold e em `tests/test_tool.py` (features recalculadas batem com o Gold).
- **Validacao temporal (rolling-origin):** 3 janelas expansivas por canal (50%, 65%, 80%), sempre treino no passado e validacao no futuro; validacao aleatoria nao e usada.
- **Hiperparametros e PCA** ajustados so dentro de cada janela de treino.
- **Estatistica:** comparacoes pareadas por canal (bootstrap e permutacao de sinais); metrica principal MAE macro por canal; tambem acuracia por pares (empate = 0,5), Spearman intra-canal, MAPE, F1 das faixas e cobertura do intervalo.
- **Pre-registro:** a partir do E-10, candidatos e regra de escolha foram escritos em `docs/07` antes de rodar; multiplas comparacoes com correcao de Holm.
- **Teste final:** aberto uma unica vez, depois de congelar o modelo (E-13).

## 7. Resultados

| Exp. | O que testou | Resultado |
| --- | --- | --- |
| E-00 | Ridge com metadados e historico | Empata com o B0 |
| E-01 a E-04 | Features de texto e agregados de historico | Sem ganho |
| E-05 | TF-IDF do texto | Promissor, nao significativo (p = 0,12) |
| E-06 | Cor, brilho e OCR da thumbnail | Sem ganho |
| E-07 | Alvo residual + perda absoluta | Melhora os modelos antigos (p = 0,018); empata com o B0 |
| **E-08** | **CatBoost + embedding SigLIP 2** | **Supera o B0: MAE -0,052 (p = 0,0006)**; controle por idade confirma sinal de conteudo |
| E-10 | PCA 8/32, combinacao, embedding de texto | Nada supera o E-08 |
| E-11 | Canal nunca visto | Ordena acima do acaso (pares 0,553, p = 0,002); nao supera o B0 em MAE |
| E-12 | LightGBM, XGBoost, ExtraTrees, ensemble, YetiRank, comparaveis, peso por canal, Optuna | Nenhum aprovado apos Holm; comparaveis promissor (p = 0,025 bruto) |
| **E-13** | **Modelo final (CB_EMB + comparaveis + rubrica Gemini) vs CB_EMB** | **Aprovado: MAE 0,850 vs 0,882 (p = 0,003, 21/26 canais)** |

Modelo anterior (CB_EMB, E-08) no rolling-origin, para referencia (o CB_FINAL tem MAE 0,850 e pares 61,7% no mesmo protocolo):

| Metrica | B0 | CB_EMB |
| --- | ---: | ---: |
| MAE em log | 0,934 | **0,882** |
| MAPE em log | 0,242 | **0,229** |
| Erro relativo mediano (curtidas) | 89% | **77%** |
| Previsao dentro de 2x do real | 55% | **58%** |
| Acuracia por pares no canal | 40% | **57-58%** |
| Spearman no canal | -0,26 | **0,20** |
| F1 macro das faixas (abaixo/media/acima/viral) | 0,11 | **0,28** |
| Cobertura do intervalo de 80% | - | **76%** (fora da amostra) |

**Teste final** (aberto uma unica vez depois da decisao do E-13; 612 Shorts mais recentes de 26 canais):

| Metrica | B0 (mediana do canal) | CB_EMB | **Modelo final (CB_FINAL)** |
| --- | ---: | ---: | ---: |
| MAE em log (macro por canal) | 0,858 | 0,780 | **0,746** |
| MAPE em log | 0,251 | 0,217 | **0,200** |
| Erro relativo mediano (curtidas) | 58% | 52% | **52%** |
| Previsao dentro de 2x do real | 58% | **64%** | 63% |
| Acuracia por pares no canal | 44% | 57% | **58%** |
| F1 macro das faixas | 0,11 | 0,30 | **0,33** |
| Cobertura do intervalo de 80% | - | - | **79,9%** |

Contra o B0: MAE -0,112 (IC95% [-0,238; -0,010], p = 0,065) e pares +14 p.p. (p < 0,001, 21 de 26 canais). Contra o CB_EMB, o ganho tem o mesmo tamanho do rolling-origin (-0,033), mas o teste (~24 Shorts por canal) nao tem poder para confirma-lo sozinho (p = 0,23).

**Onde o modelo erra** (teste final; detalhes em [docs/07](docs/07-registro-de-features.md)):

| Grupo | n | MAE modelo | MAE B0 | Cobertura 80% |
| --- | ---: | ---: | ---: | ---: |
| Moda | 324 | 0,60 | 0,66 | 86% |
| Automotivo | 288 | 0,81 | 0,97 | 73% |
| Shorts que viralizaram (> 2,7x a mediana do canal) | 93 | 1,57 | 1,88 | 47% |
| Shorts na media do canal | 171 | 0,28 | 0,12 | 99% |

- O modelo puxa a previsao para a media do canal: subestima os virais (preve viral em 6 de 93) e superestima os Shorts fracos.
- Os 10 maiores erros sao lancamentos e campanhas de marcas com mediana baixa (ex.: "Novo Jeep Avenger", 1.058 curtidas reais vs 30 previstas), provavelmente impulsionados por midia paga, que nao aparece nos dados publicos.
- Ganha do B0 em 18 de 26 canais; perde em canais com poucos Shorts no teste ou muito estaveis.

Diario completo e analises: [docs/07](docs/07-registro-de-features.md); decisoes: [docs/02](docs/02-registro-de-descobertas.md).

## 8. O que testamos e o que entrou

| Entrou no modelo | Testado e descartado (experimento) |
| --- | --- |
| Alvo residual sobre a mediana do canal + perda MAE (E-07) | Features de texto CTA/hashtags/emoji (E-01, E-02) |
| Historico e contexto do canal (E-00) | Agregados extras de historico (E-03, E-04) |
| Embedding SigLIP 2 da thumbnail, PCA 16 (E-08, E-10) | TF-IDF (E-05); embedding de texto mpnet (E-10b) |
| CatBoost (E-08, E-12) | Cor, brilho, OCR da thumbnail (E-06, E-08) |
| Intervalo conformal (tool) | PCA 8 e 32 (E-10); LightGBM, XGBoost, ExtraTrees, ensemble, YetiRank, peso por canal (E-12) |
| Comparaveis visuais + rubrica e fala do Gemini (E-13) | Destilacao por canal (D-015); modelos por canal (D-004) |

## 9. Limitacoes

- **Valor absoluto e incerto:** o intervalo de 80% cobre um fator de ~5x entre o limite inferior e o superior para criadores e ~23x para marcas; a tool serve melhor para **ordenar** criativos do mesmo canal do que para cravar um numero.
- **Virais e midia paga:** o modelo subestima Shorts que estouram (cobertura do intervalo de 47% nos virais); impulsionamento pago e distribuicao do algoritmo nao sao observaveis antes da publicacao.
- **Canal novo:** sem o canal no treino, o nivel absoluto nao supera a mediana do canal; a tool marca confianca baixa.
- **Metricas como fotografia:** curtidas medidas na data da coleta, nao em horizonte fixo; videos antigos tiveram mais tempo para acumular.
- **Recorte:** 2 setores, 26 canais, 3.004 Shorts (meta ampla do TAP: 5-15 mil e 60+ canais); beleza fora.
- **Modalidades:** thumbnail estatica + rubrica/fala do Gemini; sem frames, cortes de cena ou audio bruto (o video nao e baixado).
- **Split de teste por canal**, nao por data global.
- **Embedding opaco:** o SHAP indica que o estilo visual pesou, nao qual aspecto; comparaveis e rubrica complementam.
- **Ambiente:** modelo e agente nao rodam no Windows do Inteli (DLLs bloqueadas); o embedding ao vivo exige `torch` e acesso ao Hugging Face, e a rubrica ao vivo exige `GEMINI_API_KEY` (cada Short novo custa uma chamada, ~5 s).
- **Rubrica gerada por LLM:** as notas do Gemini sao subjetivas e podem mudar com a versao do modelo; a versao do prompt e do modelo ficam registradas no pacote.

## 10. Evolucao

1. Avaliar o modelo final em dados novos (Shorts publicados depois da coleta), ja que o teste atual foi usado.
2. Mais canais e o setor de beleza; recoleta periodica para manter o historico atual.
3. Alvo em horizonte fixo (curtidas aos 7/30 dias) com coleta recorrente.
4. Frames iniciais e audio (gancho, cortes, musica) quando houver coleta de video autorizada.
5. Dados autorizados do parceiro (YouTube Analytics via MCP: impressoes, CTR, retencao) como historico do canal (D-001).
6. Treino especialista sob demanda para um cliente com base propria (esquema minimo do TAP).
7. What-if: variacoes de titulo, horario e thumbnail re-pontuadas pelo modelo.

## 11. Documentacao

Estrutura do repositorio:

```text
src/galaxies_agent/      tool: contrato Pydantic, preditor, rubrica Gemini ao vivo, embedding, agente LangGraph, servidor MCP
scripts/coleta/          YouTube Data API, thumbnails (so biblioteca padrao) e rubrica de video + fala (Gemini)
scripts/pipeline/        Bronze -> Silver -> Feature Store -> Gold, features de thumbnail e OCR
scripts/experimentos/    E-00 a E-13, teste final (uma vez) e analise de erros
scripts/modelo/          treino e empacotamento do modelo servido pela tool
notebooks/               01_dados, 02_baseline, 03_features, 04_modelo_colab (E-08/E-10/E-12), 05_demo
tests/                   contrato, antivazamento, modelo, agente e rubrica (pytest)
data/feature_store/      features versionadas por hash (versionadas no git); bronze/silver/gold ficam fora do git
docs/                    entendimento, decisoes (D-xxx), metricas, registro de experimentos, arquitetura
```

| Doc | Conteudo |
| --- | --- |
| [01](docs/01-entendimento-do-problema.md) | Problema, recorte, decisoes |
| [02](docs/02-registro-de-descobertas.md) | Descobertas e decisoes (D-001 a D-019) |
| [03](docs/03-entendimento-dos-dados.md) | Dados |
| [04](docs/04-tabela-de-consumo.md) | Tabela de consumo |
| [05](docs/05-hipoteses-de-modelagem.md) | Hipoteses de modelagem |
| [06](docs/06-metricas-e-diagnosticos.md) | Metricas e diagnosticos |
| [07](docs/07-registro-de-features.md) | Registro de features e diario de experimentos (E-00 a E-13) |
| [08](docs/08-roadmap-de-dados-multimodais.md) | Roadmap multimodal |
| [09](docs/09-arquitetura-elt-mvp.md) | Arquitetura ELT |
| [10](docs/10-stack-e-selecao-de-modelo.md) | ADR-001: stack e selecao de modelo |

| Pasta | Notebook | Conteudo | Onde roda |
| --- | --- | --- | --- |
| `notebooks/01_dados/` | 01 | Descoberta de dados (coleta piloto, canais, volume) | local |
| | 02 | Exploracao da tabela de consumo | local |
| `notebooks/02_baseline/` | 03 | Experimentos de baseline (B0, M1-M3; executa `scripts/experimentos/run_baseline_experiments.py`) | local |
| | 04 | Metricas e diagnosticos do baseline | local |
| | 05 | Validacao rolling-origin | local |
| `notebooks/03_features/` | 06 | Analise para decisao de features | local |
| | 07 | Resultado da ablacao E-02 | local |
| `notebooks/04_modelo_colab/` | 08 | E-08: CatBoost + embedding SigLIP 2 da thumbnail | Colab (GPU) |
| | 09 | E-10b: embedding de texto | Colab |
| | 10 | E-12: selecao de modelo + Optuna + PyCaret | Colab |
| `notebooks/05_demo/` | 11 | **Demo do agente e da tool MCP** (modelo final, Gemini ao vivo) | Colab |

Os notebooks locais encontram a raiz do repo sozinhos (podem ser abertos de qualquer pasta). Os de Colab recebem um pacote `.tar.gz` arrastado para o painel Arquivos.
