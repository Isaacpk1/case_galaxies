# Roadmap de dados multimodais

## Objetivo e estado atual

Este roadmap organiza a evolucao do piloto de dados tabulares para uma base confiavel de texto, imagem, video e audio. O objetivo imediato nao e treinar um modelo multimodal grande: e tornar cada modalidade rastreavel, reutilizavel e elegivel para comparacao temporal justa.

O piloto atual possui dados brutos da YouTube Data API, uma camada Silver de Shorts elegiveis e uma tabela Gold tabular. A primeira modalidade visual e a thumbnail publica referenciada pela propria API: ela permanece associada ao mesmo `video_id`, e seu arquivo, hash e status sao registrados em Bronze. O baseline e o rolling-origin ja existem. O experimento TF-IDF E-05 deve ser considerado exploratorio; ele nao altera a arquitetura nem abre o teste final.

## Principios que guiam todas as fases

1. `video_id` e a chave de negocio que conecta metadados, midia, features, alvo e predicao.
2. Dados brutos nunca sao sobrescritos; uma nova coleta cria um novo `run_id`.
3. Arquivos de midia e features sao cacheados e versionados. Reprocessar um video so ocorre quando a fonte ou o extrator mudar.
4. A tabela Gold armazena features e referencias; nao armazena videos, imagens, audio ou vetores grandes diretamente.
5. Toda feature deve ter `available_before_publish = true` para servir ao modelo de pre-publicacao.
6. A decisao de reter uma modalidade vem do rolling-origin, por canal e por corte temporal; o teste final permanece fechado ate a configuracao escolhida.

## Arquitetura-alvo

```text
Fontes (YouTube Data API, thumbnail publica, youtube-transcript-api)
                           |
                           v
Bronze: resposta bruta + manifesto de coleta + artefato original
                           |
                           v
Silver: video elegivel + midia validada + texto/transcricao normalizada
                           |
                           v
Feature store: features por video, modalidade e versao do extrator
                           |
                           v
Gold: snapshot de treino com features, alvo, corte temporal e versoes
                           |
                           v
Rolling-origin, registro de experimento e modelo candidato
```

## Roadmap por fase

| Fase | Entrega | Criterio de saida | Nao fazer ainda |
| --- | --- | --- | --- |
| 0. Congelar contrato atual | Dicionario de campos, chaves, versao do dataset e definicao provisoria do alvo | Qualquer linha Gold pode ser rastreada ate a coleta Bronze | Alterar o alvo ou misturar novas features na tabela atual sem versao |
| 1. Manifesto e armazenamento de midia | Registro de thumbnail/video/audio por `video_id`, caminho, hash, status e data | Download idempotente e reexecutavel sem duplicar artefatos | Extrair embeddings ou treinar modelo visual |
| 2. Camada Silver multimodal | Validacao, normalizacao de texto, thumbnail e audio; tratamento de falhas | Cobertura, erros e formatos conhecidos por modalidade | Colocar arquivos binarios na tabela Gold |
| 3. Feature store versionado | Tabelas de features por modalidade, versao do extrator e disponibilidade temporal | E possivel recriar uma Gold apontando para uma versao de feature | Juntar todas as modalidades em um unico vetor sem ablation |
| 4. Definir alvo em horizonte fixo | Snapshots periodicos de metricas e `likes_d7` ou `likes_d30` | Todo video de avaliacao tem a mesma janela de acumulacao | Interpretar curtidas observadas em dias diferentes como alvo definitivo |
| 5. Modalidades isoladas | Experimentos separados: texto, thumbnail, frames, audio/transcricao | Cada modalidade tem ganho ou descarte documentado | Modelo multimodal grande de uma vez |
| 6. Fusao multimodal | Melhor combinacao de modalidades, explicabilidade e monitoramento | Supera baselines de forma consistente e cabe no custo/latencia definidos | Publicar sem teste final e sem observabilidade |

## Contratos de dados a criar

### 1. Registro de video

Uma linha por criativo conhecido. Pode ficar na Silver.

| Campo | Exemplo | Regra |
| --- | --- | --- |
| `video_id` | `abc123` | Chave unica e obrigatoria |
| `channel_id` | `UC...` | Obrigatorio |
| `published_at` | `2026-09-20T14:00:00Z` | UTC e obrigatorio |
| `source_run_id` | `20260924T163050Z` | Rastreia a coleta Bronze |
| `eligibility_status` | `eligible` | Explica entrada/saida do piloto |

### 2. Manifesto de midia

Uma linha por `video_id`, tipo e versao de download. Arquivos ficam fora do CSV.

| Campo | Regra |
| --- | --- |
| `video_id`, `media_type` | Chave composta; tipos: `thumbnail`, `video`, `audio`, `frame` |
| `source_url`, `local_path`, `content_hash` | Permitem localizar e deduplicar o artefato |
| `downloaded_at`, `extractor_version` | Permitem reproduzir a coleta |
| `status`, `error_code` | Distinguem ausencia legitima de falha de pipeline |

### 3. Registro de features

Uma linha por `video_id`, grupo e versao. Vetores grandes devem ser salvos em parquet/arquivo vetorial com uma referencia nesta tabela.

| Campo | Regra |
| --- | --- |
| `feature_set_id` | Ex.: `text_tfidf_v1`, `thumbnail_embed_v1` |
| `extractor_name`, `extractor_version` | Obrigatorios |
| `available_before_publish` | Deve ser `true` para uso no preditor |
| `created_at`, `source_hash` | Garantem rastreabilidade e invalidação de cache |
| `feature_path` ou colunas escalares | Referencia o vetor ou guarda features pequenas |

### 4. Snapshots de metricas

Uma linha por video e instante de observacao. Esta tabela e necessaria para um alvo justo.

| Campo | Regra |
| --- | --- |
| `video_id`, `observed_at` | Chave temporal |
| `like_count`, `view_count`, `comment_count` | Resultados, nunca features do proprio video |
| `age_hours` | `observed_at - published_at` |
| `target_window` | Ex.: `d7` ou `d30`, calculado apenas com idade suficiente |

## Piloto em execucao: `youtube_thumbnail`

A thumbnail publica e obtida exclusivamente da `thumbnail_url` retornada pela YouTube Data API para um Short elegivel. Nao ha download de video, audio, frames ou midia de uma plataforma externa.

```powershell
.\.venv\Scripts\python.exe scripts\collect_youtube_thumbnails.py --source-run-id 20260929T210254Z --sector automotivo --limit 10
```

Saidas esperadas por `run_id`:

- `data/bronze/thumbnails/youtube/<run_id>/files/<video_id>.<ext>`: imagem original validada;
- `data/bronze/thumbnails/youtube/<run_id>/run_summary.json`: registro operacional da coleta;
- `data/silver/thumbnail_manifest/<run_id>/thumbnail_manifest.json`: URL de origem, `video_id`, hash SHA-256, dimensoes, MIME, status e erro quando houver.

O manifesto liga a imagem ao Short por `video_id` e ao lote de metadados por `source_run_id`. Se a URL estiver ausente ou a imagem for invalida, a linha permanece no manifesto com seu status; a ausencia de thumbnail nao elimina o Short do conjunto de treino.

## Decisao para transcricoes de Shorts: `youtube-transcript-api`

Para o MVP, a camada textual de fala dos Shorts usa `youtube-transcript-api` como coletor de transcricoes automaticas. Ela recebe o mesmo `video_id` ja presente na coleta da YouTube Data API e produz texto segmentado com `start` e `duration`, sem baixar video ou audio. Isso permite testar se a fala agrega valor ao baseline de metadados, descricao, tags e thumbnail.

O coletor e executado em lotes pequenos, com manifesto por `run_id`:

```powershell
.\.venv\Scripts\python.exe scripts\collect_youtube_captions_pilot.py --video-id ID_1 ID_2 ID_3
```

Cada registro usa `transcript_status` (`downloaded`, `rate_limited`, `not_available`, `blocked` ou `failed`), idioma, quantidade de trechos, caminho e hash do JSON. Falhas de cobertura nao devem excluir a linha de treino: o modelo recebe uma flag de disponibilidade e segue com as demais modalidades.

Esta rota e experimental e deve ser monitorada por cobertura e bloqueios. O `yt-dlp` permanece apenas como fallback tecnico; nenhum dos dois coletores baixa o video.

## Ordem de implementacao recomendada

### Marco A — base de arquitetura

1. Criar os schemas e validacoes dos quatro contratos acima.
2. Definir convencao de pastas: `bronze/media/<run_id>/`, `silver/media_manifest/`, `silver/features/` e `gold/training_snapshots/`.
3. Introduzir `dataset_version` e `feature_set_ids` na Gold.
4. Documentar politica de retencao, permissao e origem de cada arquivo de midia.

**Saida:** conseguimos reproduzir o dataset atual e sabemos exatamente de que coleta e feature cada linha veio.

### Marco B — primeira modalidade operacional: thumbnail

1. Baixar e validar thumbnails usando o manifesto, com retry e cache por hash.
2. Medir cobertura, erros, resolucao e duplicatas.
3. Extrair primeiro apenas atributos baratos e interpretaveis: dimensao, proporcao, brilho, saturacao e deteccao de texto, se viavel.
4. Rodar ablation da modalidade visual contra a Gold tabular, sem testar o conjunto final.

**Saida:** pipeline de midia idempotente, com uma modalidade visual que pode ser ligada/desligada do treino.

### Marco C — texto e audio

1. Consolidar texto bruto e texto normalizado, incluindo idioma e origem.
2. Para audio, registrar licenca/permissao, formato, duracao e qualidade antes de transcrever.
3. Transcrever somente arquivos validos; guardar texto, idioma, confianca e versao do ASR.
4. Avaliar texto e audio separadamente antes da fusao.

**Saida:** cada representacao semantica e reproduzivel e possui resultado de ablation.

### Marco D — alvo e monitoramento

1. Agendar snapshots de metricas dos videos depois de publicados.
2. Escolher formalmente `likes_d7` ou `likes_d30` como alvo de produto.
3. Criar verificacoes de atraso, cobertura e mudanca de distribuicao.
4. Criar Gold versionada por horizonte do alvo e data de corte.

**Saida:** avaliacao temporal compara videos sob a mesma janela de performance.

## Decisoes pendentes antes do Marco A

| Decisao | Opcoes | Recomendacao provisoria |
| --- | --- | --- |
| Horizonte do alvo | 7, 30 ou outro numero de dias | Comecar a coletar snapshots agora; decidir entre D7 e D30 ao observar a cobertura e dinamica dos Shorts |
| Origem da midia | URL publica, upload do parceiro ou ambas | Priorizar fontes autorizadas e registrar origem/permissao |
| Armazenamento de vetores | Parquet, banco vetorial ou arquivos NumPy | Parquet/referencias locais no piloto; banco vetorial so se houver busca semantica ou escala maior |
| Prioridade de modalidade | thumbnail, texto, frames, audio | Thumbnail primeiro: menor custo e ja possui URL; depois texto; audio e frames por ultimo |
| Criterio de retencao | ganho medio ou ganho robusto | Ganho consistente em rolling-origin, sem piora material por tipo de canal e com custo aceitavel |

## Indicadores de qualidade da camada de dados

- cobertura por modalidade: percentual de `video_id` com artefato valido;
- taxa de falha e motivo de falha por extrator;
- percentual de artefatos reaproveitados do cache;
- tempo e custo por video e por lote;
- percentual de features elegiveis antes da publicacao;
- completude do alvo por horizonte (`d7`, `d30`);
- reproducibilidade: mesma versao de dados e extrator gera a mesma Gold.

## Fora de escopo agora

- Treinar um unico modelo grande sobre todas as modalidades.
- Usar metricas pos-publicacao do proprio video como feature.
- Escolher arquitetura pelo teste final atual.
- Construir banco vetorial, streaming ou microservicos antes de existir volume e caso de uso que os exija.
