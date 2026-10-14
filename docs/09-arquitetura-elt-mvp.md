# Arquitetura ELT do MVP

## Decisao

O MVP usa o `video_id` como chave de negocio para relacionar metadados, metricas, thumbnail e transcricao de um mesmo YouTube Short. A fonte de desempenho e a YouTube Data API v3; a camada de fala usa `youtube-transcript-api`; a thumbnail e processada como modalidade visual estatica.

O objetivo nao e reconstruir ou baixar videos. O objetivo e medir, com validacao temporal, se thumbnail e fala melhoram a previsao em relacao a metadados e texto.

## Fluxo

```text
                    YouTube Data API v3
                 /            |             \
       metadados/metricas  thumbnail_url    video_id
              |                  |             |
              v                  v             v
         Bronze YouTube     Bronze Thumbnail  youtube-transcript-api
                                                  |
                                                  v
                                          Bronze Transcricao
                                                  |
                                                  v
                   Silver: eligible_shorts + manifests validados
                                                  |
                                                  v
          Feature Store: contexto | texto | thumbnail | transcricao
                                                  |
                                                  v
                       Gold: snapshot temporal de treino
                                                  |
                                                  v
            Baseline + CatBoost/LightGBM + rolling-origin + SHAP
                                                  |
                                                  v
                 Tool MCP: previsao, intervalo, ranking e explicacao
```

## Contratos de dados

### Bronze

| Origem | Caminho | Chave | Conteudo |
| --- | --- | --- | --- |
| YouTube Data API | `data/bronze/youtube/<run_id>/` | `video_id` | Respostas brutas de canais e videos |
| Transcricoes | `data/bronze/transcripts/youtube_caption_pilot/<run_id>/` | `video_id` | JSON com texto, `start`, `duration` e idioma |
| Thumbnail | `data/bronze/thumbnails/<run_id>/` | `video_id` | Imagem original, quando coletada |

Bronze e imutavel. Uma nova execucao gera outro `run_id`; nenhuma coleta deve ser sobrescrita.

### Silver

| Artefato | Chave | Responsabilidade |
| --- | --- | --- |
| `eligible_shorts.csv` | `video_id` | Aplicar criterios de Short e elegibilidade |
| `transcript_manifest.json` | `video_id`, `run_id` | Registrar disponibilidade, idioma, hash, erro e caminhos |
| `thumbnail_manifest.json` | `video_id`, `run_id` | Registrar URL, dimensao, hash e status |

`transcript_status` possui cinco valores: `downloaded`, `rate_limited`, `not_available`, `blocked` e `failed`. A ausencia de uma modalidade e informacao operacional: a linha continua no dataset com uma flag de cobertura.

### Feature Store

Cada grupo de features e versionado por `feature_set_id`, `extractor_name`, `extractor_version` e `source_hash`.

| Grupo | Exemplos | Condicao |
| --- | --- | --- |
| Contexto | historico do canal, setor, horario, duracao | Disponivel antes da publicacao |
| Texto | embedding de titulo/descricao/tags, hashtags e CTA | Disponivel antes da publicacao |
| Thumbnail | embedding CLIP/SigLIP, brilho, contraste, texto em tela | Somente se thumbnail valida |
| Transcricao | embedding da fala, palavras, densidade, idioma e CTA falado | Somente se `transcript_status=downloaded` |

### Gold

Gold contem apenas referencias e colunas de features. Arquivos brutos, JSONs e vetores grandes permanecem fora dela. Cada snapshot deve informar `dataset_version`, `feature_set_ids`, data de corte e versoes de extrator.

O alvo primario e `log(1 + like_count)` ou desempenho relativo a mediana de curtidas anteriores do canal. `like_count`, `view_count` e `comment_count` do proprio Short nunca entram como feature.

## Implementacao atual (01/10/2026)

### Silver unificada

`scripts/pipeline/materialize_silver.py` agora aceita varios runs de Bronze (um por setor). O run `20261002T013000Z` junta automotivo (`20260929T210254Z`) e moda (`20260928T213206Z`): 3.189 Shorts elegiveis de 31 canais, com thumbnail validada para todos (automotivo `20260930T022637Z`, moda `20261002T005131Z`). O run anterior `20261002T003000Z` era igual, mas sem as thumbnails de moda. Cada run usa o proprio horario de coleta como corte, e o video mais recente vence quando aparece em dois runs.

```bash
python scripts/pipeline/materialize_silver.py --youtube-run-id 20260929T210254Z 20260928T213206Z --thumbnail-run-id 20260930T022637Z 20261002T005131Z --silver-run-id 20261002T013000Z
# consolidate_transcript_manifest.py (legendas): removido em 14/10/2026 na limpeza do repo; continua no historico do git; a fala agora vem do Gemini
```

### Coleta no Windows do Inteli

A politica de Controle de Aplicativos bloqueia DLLs nativas (pandas, numpy, Pillow, PaddleOCR). Por isso os coletores que rodam no Windows usam so a biblioteca padrao do Python: `scripts/coleta/collect_youtube_thumbnails_stdlib.py` e `scripts/coleta/collect_gemini_video_features.py` (o antigo coletor de legendas foi removido em 14/10/2026 na limpeza do repo; continua no historico do git) (urllib no lugar de requests, csv no lugar de pandas). Validacao de imagem, features e OCR rodam fora do Windows. As thumbnails de moda foram coletadas assim em 02/10 (1.683/1.683).

### Feature Store

Codigo: `scripts/pipeline/feature_store.py` (registro e regras) e `scripts/pipeline/build_feature_store.py` (extratores).

```text
data/feature_store/registry.json
data/feature_store/<nome>/<feature_set_id>/features.csv     uma linha por video_id
data/feature_store/<nome>/<feature_set_id>/metadata.json   versao, fontes, hash, cobertura, docs das colunas
```

`feature_set_id = <nome>__<hash[:10]>`, onde o hash cobre nome, extrator, versao, parametros e o sha256 de cada arquivo de entrada. Mesma entrada e mesmo codigo geram o mesmo id; qualquer mudanca gera um id novo, sem sobrescrever. A escrita recusa `video_id` duplicado e colunas `like_count`, `view_count`, `comment_count` ou `target_*`.

| feature_set_id | Grupo | Colunas | Cobertura | Observacao |
| --- | --- | --- | --- | --- |
| `context_v1__cfd179d115` | A. Metadados e contexto | 25 | 100% | Historico estritamente anterior; inclui `baseline_log_likes` (B0) |
| `text_v1__aa3925afc1` | B. Texto | 15 | 100% | T-01 a T-11; embedding fica para `text_v2` |
| `thumbnail_v1__5d64f40762` | C. Thumbnail | 21 | 100% | Visual + OCR, moda e automotivo |
| `transcript_v1__1b208c6972` | D. Transcricao | 13 | 10% | 323 legendas; ver D-012 sobre traducoes |

```bash
python scripts/pipeline/build_feature_store.py --silver-run-id 20261002T013000Z
```

Ainda nao implementado do desenho no Miro: inscritos do canal (o valor da API e de hoje, nao da data do post, entao vazaria futuro), embedding textual, embedding CLIP/SigLIP, rosto e produto em destaque, embedding da fala. Os hosts de modelos (Hugging Face) estao bloqueados nos ambientes atuais.

### Gold

Codigo: `scripts/pipeline/build_gold_snapshot.py`. Saida imutavel em `data/gold/snapshots/<dataset_version>/` com `training_snapshot.csv` e `manifest.json`; `data/gold/snapshots/LATEST.json` aponta para a versao atual.

Snapshot atual: `gold_v1_20261002T013000Z_663f502d` (o anterior `gold_v1_20261002T003000Z_fa5d11d1` nao tinha thumbnail de moda), com 3.004 Shorts de 26 canais (moda 1.589, automotivo 1.415), treino 2.392 e teste 612, e 74 colunas de features.

- Canais: `meets_minimum_history` (>= 30 Shorts elegiveis). Linhas: `channel_prior_posts >= 5`.
- Alvos: `target_log_likes` (principal), `target_residual_log_likes` (desvio sobre o B0, usado no E-07) e `target_relative_to_prior_median`.
- Split: os 20% mais recentes de cada canal sao teste; posts no mesmo instante ficam do mesmo lado.
- Checagens automaticas (o build falha se alguma nao passar): nenhuma metrica do proprio video como feature; treino antes do teste em todo canal; B0 recalculado da Silver para 300 linhas so com posts anteriores (erro maximo 0); todos os feature sets com `available_before_publish = true`.
- Consistencia com o Gold piloto (`data/gold/pilot_consumption_table.csv`): 2.995 videos em comum, 99,7% com o mesmo split, B0 rolling-origin 0,9338 contra 0,9346. As diferencas vem da coleta de automotivo um dia mais recente. O Gold piloto fica intacto para reproduzir E-00 a E-07.

```bash
python scripts/pipeline/build_gold_snapshot.py --silver-run-id 20261002T013000Z
```

## Decisoes e trade-offs

| Decisao | Motivo | Consequencia |
| --- | --- | --- |
| YouTube Data API para desempenho | Origem oficial de metadados e metricas | Nao fornece o arquivo de video |
| `youtube-transcript-api` para fala | Retorna texto e timestamps sem baixar video | Cobertura e estabilidade nao sao garantidas |
| Apify como provider alternativo de legendas | IPs locais bloqueados pelo YouTube (D-011) | Fornecedor terceiro, custo por transcricao e `usage_note` proprio; chamado via REST com `requests` (o `apify-client` usa DLL bloqueada no Windows do Inteli) |
| Thumbnail antes de frames | Menor custo e maior disponibilidade | Representa apenas uma imagem estatica |
| Modelos tabulares sobre features | Auditavel, rapido e comparavel ao baseline | Nao substitui modelos de video completo |
| Rolling-origin | Simula previsao no futuro | Avaliacao mais cara e conservadora |

## Medicao do ganho multimodal

| Experimento | Feature sets adicionados | Pergunta respondida |
| --- | --- | --- |
| E-01 | contexto | O historico do canal supera a mediana? |
| E-02 | contexto + texto | A mensagem publicada melhora a previsao? |
| E-03 | E-02 + thumbnail | A imagem estatica adiciona sinal? |
| E-04 | E-03 + transcricao | A fala adiciona sinal alem do texto publicado? |

Todos os experimentos usam o mesmo corte temporal. A decisao de manter uma modalidade exige ganho consistente em MAE log, Spearman e acuracia par-a-par, sem piora relevante por setor ou tipo de canal.

## Selecao de algoritmo e hiperparametros

PyCaret pode ser usado apenas para uma comparacao exploratoria de poucos modelos. A configuracao que avanca para o teste final deve ser produzida por scripts explicitos, com CatBoost como candidato principal e busca aleatoria dentro dos folds rolling-origin. Grid search exaustivo nao e recomendado para o MVP, pois consome muitas combinacoes sem garantir melhor cobertura do espaco.

O periodo futuro reservado nunca participa da escolha de hiperparametros. A configuracao e congelada antes do teste final. A decisao completa de stack, alternativas e espaco inicial de busca esta na [ADR-001](10-stack-e-selecao-de-modelo.md).

## Operacao e falhas esperadas

1. Coletar primeiro os metadados da API e materializar `eligible_shorts`.
2. Selecionar um lote pequeno de `video_id` para a transcricao.
3. Registrar todos os resultados no manifesto, inclusive falhas.
4. Extrair features somente de artefatos com status valido.
5. Fazer o join pelo `video_id` e manter flags de cobertura.

Se a transcricao estiver ausente, o pipeline segue com texto, thumbnail e contexto. A coleta local nao usa cookies, contas nem proxies e nunca baixa video. Quando a coleta e feita pelo provider `apify`, a origem terceirizada fica registrada em cada record (`provider`, `apify_actor`, `apify_run_id`, `usage_note`).

## Saida da tool

A tool recebe um criativo novo, legenda planejada, horario, setor e contexto do canal. Ela devolve previsao, intervalo, desempenho relativo ao canal, ranking, `top_drivers`, explicacao e `confidence_flag`.

Para um criativo ainda nao publicado, apenas features disponiveis antes da publicacao podem ser usadas. A tool deve reduzir a confianca quando faltar historico do canal, thumbnail ou transcricao.
