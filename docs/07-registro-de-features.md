# Registro de features e experimentos

## Objetivo

Este arquivo e o catalogo de variaveis do modelo. Cada feature tem uma definicao verificavel, a fonte de dados, uma regra de disponibilidade antes da publicacao e um status de experimento.

O objetivo nao e acumular colunas: uma feature so passa a fazer parte do modelo candidato se melhorar de forma consistente a validacao rolling-origin. O teste final permanece fechado ate a escolha da configuracao.

## Status possiveis

| Status | Significado |
| --- | --- |
| `baseline` | Ja usada no modelo de referencia atual. |
| `candidata` | Definida, mas ainda nao calculada ou avaliada. |
| `em_teste` | Incluida no proximo experimento rolling-origin. |
| `retida` | Apresentou ganho consistente e segue para a proxima configuracao candidata. |
| `sem_ganho` | Foi testada, mas nao melhorou as metricas de modo consistente. |
| `descartada` | Nao pode ser usada por vazamento, baixa qualidade ou custo desproporcional. |

## Regra antileakage

Nenhuma feature pode usar `like_count`, `view_count` ou `comment_count` do video previsto. Essas metricas sao permitidas apenas como agregados de videos publicados **antes** dele. Todo calculo de historico deve respeitar `published_at` do proprio canal.

## Features atuais do baseline

| Grupo | Feature | Fonte | Status | Observacao atual |
| --- | --- | --- | --- | --- |
| Historico | `baseline_log_likes` | Mediana de curtidas dos posts anteriores | `baseline` | Maior importancia por permutacao; esperado para previsao personalizada por canal. |
| Historico | `channel_prior_posts` | Contagem de posts anteriores | `baseline` | Mede quantidade de evidencia disponivel. |
| Historico | `days_since_previous_post` | Diferenca para a publicacao anterior | `baseline` | Mede cadencia de postagem. |
| Formato | `duration_seconds` | Metadado do Short | `baseline` | Teve sinal preditivo na primeira rodada. |
| Texto resumido | `title_length` | Titulo | `baseline` | Sem ganho isolado conclusivo ate agora. |
| Texto resumido | `description_length` | Descricao | `baseline` | Sem ganho isolado conclusivo ate agora. |
| Texto resumido | `tag_count` | Tags | `baseline` | Sem ganho isolado conclusivo ate agora. |
| Contexto | `publish_hour_utc`, `publish_weekday` | Data de publicacao | `baseline` | Contexto de horario; sera reavaliado com versoes ciclicas. |
| Contexto | `channel_id`, `sector`, `channel_type` | Catalogo de canais | `baseline` | Representam contexto do canal; nao validam generalizacao para canais novos. |

## Candidatas: texto e metadados

| ID | Feature ou grupo | Definicao | Fonte | Status |
| --- | --- | --- | --- | --- |
| T-01 | `title_word_count` | Quantidade de palavras no titulo | `title` | `candidata` |
| T-02 | `description_word_count` | Quantidade de palavras na descricao | `description` | `candidata` |
| T-03 | `has_question` e `exclamation_count` | Presenca de `?` e quantidade de `!` | titulo + descricao | `candidata` |
| T-04 | `emoji_count` | Quantidade de caracteres emoji | titulo + descricao | `candidata` |
| T-05 | `hashtag_count`, `mention_count`, `url_count` | Ocorrencias de `#`, `@` e URLs | titulo + descricao | `candidata` |
| T-06 | `digit_count` e `uppercase_ratio` | Numeros e proporcao de letras maiusculas | titulo + descricao | `candidata` |
| T-07 | flags de CTA | Indicadores de termos como `curta`, `comente`, `compartilhe`, `salve` e `veja ate o fim` | titulo + descricao | `candidata` |
| T-08 | `title_tag_overlap` | Proporcao de termos do titulo tambem presentes nas tags | titulo + tags | `candidata` |
| T-09 | duracao em faixas | Categorias: ate 15s, 16-30s, 31-60s, 61-180s | `duration_seconds` | `candidata` |
| T-10 | tempo ciclico | `sin` e `cos` da hora e do dia da semana | `published_at` | `candidata` |
| T-11 | fim de semana | Indicador de sabado/domingo | `published_at` | `candidata` |

## Candidatas: historico do canal

| ID | Feature ou grupo | Definicao | Fonte | Status |
| --- | --- | --- | --- | --- |
| H-01 | media e desvio de curtidas anteriores | Media e desvio padrao acumulados antes do video | `like_count` historico | `candidata` |
| H-02 | mediana recente | Mediana dos 5 e 10 Shorts anteriores | `like_count` historico | `candidata` |
| H-03 | tendencia recente | Inclinacao das curtidas em log nos 5 ou 10 posts anteriores | `like_count` historico | `candidata` |
| H-04 | volatilidade recente | Desvio ou intervalo interquartil dos 5 ou 10 posts anteriores | `like_count` historico | `candidata` |
| H-05 | views e comentarios anteriores | Mediana/medio de views e comentarios de posts anteriores | metricas historicas | `candidata` |
| H-06 | taxa historica de curtidas | Mediana de `(likes + 1) / (views + 1)` nos posts anteriores | metricas historicas | `candidata` |
| H-07 | frequencia recente | Posts nos ultimos 7/30 dias anteriores e cadencia media | `published_at` historico | `candidata` |

### Resultado do lote E-01

T-01 a T-11 foram calculadas e testadas juntas como um lote de texto e metadados. O lote reduziu o MAE no teste final pontual, mas piorou o MAE medio do rolling-origin em M1, M2 e M3. Logo, o resultado e `sem_ganho` para o **lote coletivo**; ele nao permite concluir que cada feature individual seja inutil.

Entre as features do lote, `has_cta`, `title_tag_overlap` e `duration_bucket` apresentaram dependencia preditiva no M1, mas essa importancia nao foi suficiente para justificar reter o lote inteiro. As variaveis permanecem catalogadas para futuros testes de ablation em grupos menores.

### Resultado da ablation E-02

E-02 testou somente `has_cta`, `title_tag_overlap` e `duration_bucket`. Para M2, o MAE medio do rolling-origin caiu de 0,7615 para 0,7500, mas a comparacao pareada por canal E-02 - E-00 foi -0,0115 com IC bootstrap de 95% [-0,0351, 0,0082] e p-valor de permutacao 0,3784. Seis canais melhoraram e seis pioraram.

O lote reduzido e mais promissor que E-01, mas ainda e `sem_ganho` para decisao de produto: a evidencia nao confirma melhora consistente. As tres variaveis continuam candidatas para reavaliacao quando houver mais canais, mais observacoes ou features historicas adicionais.

## Itens para fase posterior

| ID | Feature ou grupo | Dependencia | Status |
| --- | --- | --- | --- |
| M-01 | TF-IDF de titulo, descricao e tags | Vetorizacao unigramas/bigramas, no maximo 500 termos; dados disponiveis antes da publicacao | `sem_ganho` |
| M-02a | atributos visuais da thumbnail (`thumbnail_v1`) | Thumbnails validadas na Silver; ver secao abaixo | `sem_ganho` (E-06) |
| M-02b | texto na thumbnail via OCR (`thumbnail_ocr_v1`) | EasyOCR sobre o painel de conteudo; ver secao abaixo | `sem_ganho` (E-06) |
| M-02d | embedding SigLIP 2 da thumbnail (`google/siglip2-base-patch16-224`, PCA 16) | Thumbnail com painel vertical recortado | `retida` (E-08) |
| M-02c | frames do video | Download/cache de midia | `candidata` |
| M-03 | transcricao, fala e audio | Extracao de audio e ASR | `candidata` |

## Thumbnail: `thumbnail_v1` e `thumbnail_ocr_v1` (2026-10-01)

Fonte: `data/silver/thumbnail_manifest/20260930T025500Z` (1.506 thumbnails validas, todas de canais automotivos). Ambas as features usam somente a thumbnail publicada, entao estao disponiveis antes das metricas de engajamento e nao violam a regra antileakage.

**Descoberta importante: pillarbox.** Cerca de 62% das thumbnails sao um frame vertical 9:16 no centro com copias desfocadas e escurecidas nas laterais (96% nos canais `creator`, 46% nos `brand`). Calcular brilho/cor na imagem inteira enviesava tudo para escuro (brilho medio 0,26 -> 0,37 apos o recorte) e o OCR lia o mesmo texto duas vezes. Os dois scripts detectam o painel central (laterais com brilho < 50% do centro, ou < 65% com borda vertical nitida) e calculam tudo apenas nele. Regra calibrada visualmente nas 1.506 imagens.

| Feature | Definicao | Script / saida |
| --- | --- | --- |
| `is_vertical_pillarbox` | 1 se a thumbnail e frame vertical com laterais desfocadas | `scripts/extract_thumbnail_features.py` -> `data/silver/features/thumbnail_v1/20261001T222256Z/` |
| `letterbox_fraction`, `is_low_resolution` | Area de barras pretas; thumbnail sem `maxresdefault` (480x360) | idem |
| `brightness_mean`, `brightness_std` | Luminancia media e contraste no painel de conteudo | idem |
| `saturation_mean`, `colorfulness` | Saturacao HSV media; colorfulness de Hasler & Suesstrunk | idem |
| `dark_pixel_pct`, `light_pixel_pct` | % de pixels com luma < 0,2 e > 0,8 | idem |
| `edge_density` | % de pixels com gradiente forte (imagem "carregada") | idem |
| `dominant_hue` | Faixa de cor dominante ou `neutral` | idem |
| `has_text`, `ocr_word_count`, `ocr_char_count` | Ha texto (>= 3 caracteres) e quanto texto | `scripts/extract_thumbnail_ocr.py` -> `data/silver/features/thumbnail_ocr_v1/20261001T221500Z/` |
| `ocr_text_area_pct` | Fracao da area do painel coberta por caixas de texto | idem |
| `ocr_uppercase_ratio`, `ocr_has_digits`, `ocr_has_question`, `ocr_has_exclamation` | Estilo do texto sobreposto | idem |
| `ocr_mean_confidence`, `ocr_text` | Confianca media e texto bruto (para TF-IDF/embedding depois) | idem |

**OCR.** PaddleOCR foi descartado neste ambiente: exige Python <= 3.12 e a politica de aplicativos do Windows do Inteli bloqueia suas DLLs. Foi usado EasyOCR (`latin_g2`, pt+en, CPU), rodado fora da maquina e salvo na Silver. Caixas com confianca < 0,40 ficam no `ocr_raw.jsonl` mas fora das features. Resultado: 68% das thumbnails tem texto (83% `creator`, 62% `brand`), mediana de 6 palavras quando ha texto. O texto tem ruido (letras estilizadas, fontes grandes), entao usar primeiro as contagens/flags; `ocr_text` so como entrada de TF-IDF com cautela.

**Sinal bruto (Spearman com log de curtidas, sem controle de canal):** `edge_density` 0,24, `is_vertical_pillarbox` 0,22, `ocr_uppercase_ratio` 0,40, `ocr_has_question` 0,20; demais |rho| <= 0,11. Atencao: essas variaveis separam `creator` de `brand`, entao boa parte desse sinal pode ser efeito de canal ja capturado pelo baseline. So o experimento rolling-origin (E-06) decide.

**Limitacao:** a Silver `eligible_shorts` atual so tem automotivo; os canais de moda ainda precisam entrar no manifesto de thumbnails para que as features cubram os dois setores.

### Resultado do E-06 (thumbnail)

Script: `scripts/run_thumbnail_experiment.py`; artefatos em `artifacts/feature_experiments/e06_thumbnail/`. Mesmo protocolo do E-00 a E-05 (Ridge M1/M2/M3, rolling-origin com 3 origens, alpha escolhido em holdout cronologico, bootstrap e permutacao exata pareados por canal). Escopo restrito a automotivo (unico setor com thumbnails na Silver); os modelos de referencia sem thumbnail foram reajustados nas mesmas linhas.

| Comparacao | Delta MAE macro | IC 95% bootstrap | p (permutacao) | Canais melhor / pior |
| --- | --- | --- | --- | --- |
| M1 + visual | -0,028 | [-0,072; 0,010] | 0,22 | 10 / 4 |
| M1 + OCR | -0,017 | [-0,050; 0,019] | 0,38 | 11 / 3 |
| M1 + ambos | -0,031 | [-0,074; 0,010] | 0,19 | 10 / 4 |
| M3 + ambos | -0,040 | [-0,085; 0,005] | 0,11 | 10 / 4 |
| M2 + ambos | -0,008 | [-0,039; 0,023] | 0,63 | 6 / 8 |

Leitura: as features de thumbnail reduzem o erro de M1 e M3 nas tres origens e na maioria dos canais, em `brand` e em `creator`, mas o intervalo de confianca ainda inclui zero. O ganho vem principalmente dos atributos visuais; o OCR sozinho ajuda menos. Em M2 (modelo por tipo de canal) nao ha ganho. Status: `sem_ganho`, mas e o sinal mais consistente desde o E-05.

**Alerta mais importante:** no recorte automotivo, o B0 (mediana historica do canal) tem MAE 1,092 e nenhum Ridge o supera (melhor: M2+ambos com 1,130; M1 fica 0,090 pior que B0, p<0,001, pior em 13 de 14 canais). O R2 medio por canal e negativo em todos os modelos. Antes de somar mais features, vale revisar a formulacao: prever o desvio em relacao ao B0 (`target_log_likes - baseline_log_likes`) em vez do nivel, ou usar `baseline_log_likes` como offset.

### Resultado do E-07 (formulacao residual + LAD)

Script: `scripts/run_residual_experiment.py --scope {automotivo,all} --loss {ridge,lad}`; artefatos em `artifacts/feature_experiments/e07_residual_<loss>_<scope>/`.

**Por que o B0 ganhava de todos os modelos.** Duas causas somadas: (1) o Ridge prevê o nivel de curtidas do zero, entao gasta capacidade reaprendendo o tamanho de cada canal; (2) o Ridge minimiza erro quadratico e portanto prevê a **media**, enquanto o MAE (metrica principal) premia a **mediana**. Como os residuos em log sao assimetricos (poucos Shorts virais), a media fica deslocada e perde MAE para o B0, que ja e uma mediana. Sinal disso: B0 vencia em MAE mas perdia em RMSE.

**O que mudou:** o alvo passa a ser `target_log_likes - baseline_log_likes` (o modelo aprende so o desvio "acima ou abaixo do normal do canal") e a perda passa a ser absoluta (LAD, via minimos quadrados iterativamente reponderados com a mesma penalidade ridge).

| Comparacao (todos os setores, 26 canais) | Delta MAE | IC 95% | p | Canais melhor / pior |
| --- | --- | --- | --- | --- |
| R_M2 (residual+LAD) vs M2 (antigo) | -0,052 | [-0,090; -0,012] | 0,018 | 21 / 5 |
| R_M1 vs M1 | -0,055 | [-0,090; -0,021] | 0,003 | 21 / 5 |
| R_M2 vs B0 | -0,007 | [-0,025; 0,009] | 0,45 | 12 / 14 |

| Comparacao (automotivo, 14 canais) | Delta MAE | IC 95% | p | Canais melhor / pior |
| --- | --- | --- | --- | --- |
| R_M2 + thumbnail vs R_M2 | -0,019 | [-0,047; 0,007] | 0,22 | 10 / 4 |
| R_M1 + thumbnail vs R_M1 | +0,005 | [-0,010; 0,021] | 0,57 | 8 / 6 |
| R_M2 + thumbnail vs B0 | -0,015 | [-0,050; 0,015] | 0,47 | 7 / 7 |

Leitura:
1. A nova formulacao e uma melhora real e significativa sobre os modelos antigos. Fica `retida` como nova referencia (R_M2).
2. Contra o B0 ela empata: vence nas 3 origens, mas a margem e pequena e nao significativa. Ou seja, as features atuais quase nao trazem informacao alem do "normal do canal".
3. A melhora da thumbnail no E-06 vinha em boa parte de corrigir o vies do modelo antigo; na nova formulacao o ganho some em M1/M3 e fica fraco em M2. Thumbnail continua `sem_ganho`.
4. A acuracia de ranking dentro do canal fica em torno de 0,52 (0,5 = moeda). Prever qual criativo vai performar melhor so com metadados e thumbnail ainda e dificil; isso reforca a prioridade de transcricao, moda nas features multimodais e mais canais.

### Resultado do E-08 (CatBoost residual + embedding SigLIP 2 da thumbnail)

Notebook: `notebooks/08_e08_catboost_residual_multimodal.ipynb` (Colab, GPU T4). Artefatos: `artifacts/feature_experiments/e08_catboost_residual/`. Gold `gold_v1_20261002T013000Z_663f502d`, 2.392 linhas de treino, 26 canais, moda e automotivo. Mesmo rolling-origin do E-07; hiperparametros por busca aleatoria (12 tentativas) em holdout cronologico dentro de cada janela; PCA (16 dim) do embedding ajustado so no treino de cada janela. Teste final fechado.

| Modelo | MAE log | Pares (justo) | Spearman intra-canal | Origens vencendo B0 |
| --- | ---: | ---: | ---: | ---: |
| CB_EMB (BASE + embedding SigLIP 2) | **0,8815** | 0,569 | 0,198 | 3/3 |
| CB_FULL (BASE + thumbnail manual + texto E-02 + embedding) | 0,8877 | **0,584** | **0,230** | 3/3 |
| CB_THUMB (BASE + thumbnail manual/OCR) | 0,9210 | 0,547 | 0,119 | 2/3 |
| CB_BASE (CatBoost, so BASE) | 0,9258 | 0,510 | 0,032 | 2/3 |
| R_M2 (referencia E-07) | 0,9279 | 0,534 | 0,096 | 3/3 |
| B0 | 0,9338 | 0,402 | -0,263 | - |

| Comparacao (MAE, 26 canais) | Delta | IC 95% | p | Canais melhor |
| --- | ---: | --- | ---: | ---: |
| CB_EMB vs B0 | -0,052 | [-0,081; -0,025] | 0,0006 | 18 |
| CB_EMB vs R_M2 | -0,046 | [-0,072; -0,025] | 0,0002 | 21 |
| CB_EMB vs CB_BASE | -0,044 | [-0,066; -0,023] | 0,0005 | 21 |
| CB_THUMB vs CB_BASE | -0,005 | [-0,028; 0,018] | 0,70 | 15 |
| CB_BASE vs R_M2 | -0,002 | [-0,031; 0,027] | 0,89 | 12 |

Leitura:
1. Primeiro resultado que supera o B0 com significancia. O ganho vem do **embedding da thumbnail**, nao do algoritmo: CatBoost sem conteudo empata com o R_M2. No SHAP do modelo final, o grupo `thumb_embedding (PCA)` e de longe o maior, seguido de `duration_seconds` e `baseline_log_likes`.
2. Features manuais da thumbnail (cor, brilho, OCR) seguem `sem_ganho`, coerente com E-06.
3. Ranking: acuracia por pares justa (empate previsto = 0,5) sobe de 0,534 (R_M2) para 0,57-0,58. O criterio antigo punia empates e deixava o B0 em 0,26; no criterio justo ele fica em 0,40.
4. Intervalo de 80% (CatBoost MultiQuantile) cobre so 50-53%: estreito demais. Precisa de calibracao (predicao conformal sobre os residuos do rolling-origin) antes de ir para a tool.
5. **Controle de idade** (diagnostico, sem retreino): Spearman idade x curtidas dentro do canal e fraco (-0,06). Com o alvo ajustado pela idade do video na coleta, CB_EMB mantem Spearman 0,233 contra 0,124 do CB_BASE (delta +0,11, IC 95% [0,03; 0,19], p=0,013, 17 de 25 canais); a previsao do CB_EMB quase nao correlaciona com a idade (0,05). O sinal do embedding e de conteudo, nao de recencia.
6. O Spearman negativo do B0 dentro do canal nao e efeito de idade; o mais provavel e regressao a media (a mediana movel sobe depois de uma sequencia de videos fortes, e os seguintes tendem a voltar ao normal).

Status: embedding `retido`; CB_EMB vira a nova referencia. Proximo: calibrar intervalo, somar rubrica Gemini (E-09) e so entao abrir o teste final uma vez.

### Metricas complementares do TAP (E-08, rolling-origin)

Arquivo: `artifacts/feature_experiments/e08_catboost_residual/mape_f1_bands.json`. Faixas pelo desvio sobre a mediana do canal (log): abaixo < -0,3 <= na media <= 0,3 < acima <= 1,0 < viral. Distribuicao real: abaixo 37%, na media 29%, acima 20%, viral 14%.

| Modelo | MAPE (log) | Erro relativo mediano (curtidas) | Dentro de 2x | F1 macro das faixas | Acuracia da faixa |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0,242 | 89% | 55% | 0,11 | 29% |
| R_M2 | 0,237 | 83% | 56% | 0,23 | 32% |
| CB_EMB | **0,229** | **77%** | **58%** | **0,28** | **35%** |

MAPE em escala log exclui 10 Shorts com 0 curtida (denominador zero). O B0 preve sempre "na media", dai o F1 baixo. O CB_EMB raramente preve "viral" (6 de 173 virais acertados): o modelo encolhe para o centro, como esperado com perda MAE; a faixa deve ser comunicada como tendencia, nao como garantia.

### Protocolo pre-registrado: E-10 e E-11 (escrito antes de rodar, 12/10/2026)

Referencia: CB_EMB do E-08 com os hiperparametros finais fixos (sem nova busca, para nao abrir mais graus de liberdade). Mesmo Gold, mesmo rolling-origin, teste final fechado.

- **E-10a, dimensoes do PCA do embedding da thumbnail**: 8, 16 (referencia) e 32.
- **E-10b, embedding de texto** (`paraphrase-multilingual-mpnet-base-v2`, PCA 16): CB_EMB + texto(titulo) e CB_EMB + texto(titulo + descricao).
- **E-10c, combinacao**: CB_EMB + features manuais de thumbnail + texto E-02 (o CB_FULL do E-08) com os mesmos parametros fixos.
- **Regra de escolha**: troca a referencia so se o MAE macro melhorar com p < 0,05 na permutacao pareada por canal contra o CB_EMB de 16 dimensoes; empate em MAE (p >= 0,05) e desempatado pela acuracia por pares justa so se a diferenca de pares tambem tiver p < 0,05. Caso contrario, fica o CB_EMB de 16. Sao 5 comparacoes; resultados com p entre 0,01 e 0,05 serao lidos com cautela (multiplas comparacoes).
- **E-11, canal nao visto**: para cada canal e cada origem, treina com as janelas de treino dos OUTROS canais e preve a janela de validacao do canal deixado de fora. O B0 do canal continua vindo do historico publico dele; `channel_id` sai das features. Modelos: B0, CB_BASE e CB_EMB (parametros fixos). Pergunta: o ganho do embedding sobre o B0 se mantem em canais que o modelo nunca viu? Nao escolhe modelo; mede generalizacao.

### Resultado do E-10a/E-10c (PCA e combinacao) e do E-11 (canal nao visto)

Script: `scripts/run_e10_e11_experiments.py` (retomavel); artefatos em `artifacts/feature_experiments/e10_e11/` (`report.json`). Parametros fixos do CB_EMB do E-08.

**E-10a/c** (rolling-origin, 26 canais):

| Braco | MAE | Pares (justo) | vs CB_EMB_pca16 (MAE) |
| --- | ---: | ---: | --- |
| CB_EMB_pca16 (referencia) | 0,8824 | 0,581 | - (vs B0: -0,052, p=0,008, 21/26 canais) |
| CB_FULL_fixed | 0,8836 | 0,579 | +0,001, p=0,86 |
| CB_EMB_pca32 | 0,8879 | 0,571 | +0,006, p=0,57 |
| CB_EMB_pca8 | 0,8885 | 0,566 | +0,006, p=0,50 |

**E-10b** (embedding de texto `paraphrase-multilingual-mpnet-base-v2`, PCA 16, gerado no Colab com `notebooks/09_e10b_text_embedding.ipynb`; Silver `data/silver/features/text_mpnet_v1/20261013T020000Z/`):

| Braco | MAE | Pares (justo) | vs CB_EMB_pca16 |
| --- | ---: | ---: | --- |
| CB_EMB + texto (titulo) | 0,8783 | 0,568 | MAE -0,004 (p=0,73); pares -0,013 (p=0,46) |
| CB_EMB + texto (titulo + descricao + tags) | 0,8767 | 0,593 | MAE -0,006 (p=0,58); pares +0,012 (p=0,57) |

Pela regra pre-registrada, nenhum braco substitui a referencia: **fica o CB_EMB com PCA 16**. As features manuais/texto do CB_FULL nao acrescentam nada com parametros fixos (a vantagem de ranking vista no E-08 nao se repete).

**E-11** (deixa um canal fora; o modelo nunca viu o canal, `channel_id` fora das features; B0 do historico publico do canal):

| Modelo | MAE | Pares (justo) |
| --- | ---: | ---: |
| B0 | 0,9338 | 0,402 |
| CB_BASE_loco (sem conteudo) | 1,0103 | 0,510 |
| CB_EMB_loco (com embedding) | 0,9508 | **0,553** |

- MAE: em canal nao visto o modelo **nao supera o B0** (+0,017, IC 95% [-0,014; 0,052], p=0,36). O ajuste fino ao nivel de cada canal depende de ter visto o canal no treino.
- Ranking dentro do canal: o CB_EMB ordena acima do acaso em canais nunca vistos: 0,553 de pares (IC 95% [0,522; 0,580] contra 0,5; p=0,002; 22 de 26 canais acima de 0,5). Sem embedding fica no acaso (0,510, p=0,53).
- O embedding melhora o modelo tambem fora da amostra de canais: CB_EMB_loco vs CB_BASE_loco, MAE -0,060 (p=0,003, 18/26) e pares +0,042 (p=0,048).

Leitura para o produto: para um cliente novo (canal fora da base) a tool serve para **ordenar criativos do mesmo canal** com sinal modesto e real; a previsao absoluta de curtidas nao e melhor que a mediana do canal. A tool ja marca `confidence_flag = baixa` para canais fora da base.

### Protocolo pre-registrado: E-12 (algoritmos, objetivo de ranking, comparaveis, Optuna) - 12/10/2026

Notebook: `notebooks/10_e12_model_selection.ipynb` (Colab). Mesmo Gold, mesmo rolling-origin, teste final fechado. Todos os bracos preveem o desvio sobre o B0 com as mesmas features do CB_EMB (BASE + PCA 16 do SigLIP 2, PCA ajustado em cada janela), salvo onde indicado. Bracos com busca aleatoria usam 10 tentativas no holdout cronologico interno de cada janela (mesmo procedimento que gerou os parametros do CB_EMB).

Candidatos (7): `LGBM` (LightGBM, perda L1), `XGB` (XGBoost, perda absoluta), `ET` (ExtraTrees), `ENS` (media CB_EMB + LAD linear com embedding), `CB_RANK` (CatBoost YetiRank agrupado por canal; nivel vem do CB_EMB, ordem dentro do canal vem do ranking), `CB_KNN` (CB_EMB + desvio medio dos 5 Shorts anteriores mais parecidos do mesmo canal + novidade visual em relacao ao canal), `CB_W` (CB_EMB com peso igual por canal).

Regra de escolha: um candidato substitui o CB_EMB se o MAE macro melhorar com p < 0,05 apos correcao de Holm sobre os 7 candidatos (permutacao pareada por canal). Para `CB_RANK` o criterio e a acuracia por pares justa (mesma correcao). Se nenhum passar, fica o CB_EMB.

Optuna (fase 2, so no vencedor ou no CB_EMB): 60 tentativas otimizando o MAE macro medio das origens 1 e 2; a configuracao escolhida e avaliada so na origem 3 contra a referencia (comparacao sem uso na escolha). Troca apenas se melhorar na origem 3 com p < 0,05.

PyCaret: so exploratorio (ADR-001), com o split temporal; nao entra na regra de escolha.

### Protocolo pre-registrado: E-13 (modelo final) - 12/10/2026

Uma unica comparacao, escrita antes de rodar e depois de E-08 a E-12, para consolidar o modelo sem nova rodada de selecao.

- **Referencia:** CB_EMB (E-08), parametros do E-08 (ou os do Optuna do E-12, so se ele tiver sido aprovado na origem 3).
- **Candidato unico CB_FINAL:** CB_EMB + comparaveis (`knn5_resid_mean`, `knn5_sim_mean`, `visual_novelty`, do E-12) + rubrica Gemini (notas 0-10, flags e `tom_narracao` como categoria) + fala derivada da transcricao Gemini (`tem_fala`, palavras por segundo). Videos sem rubrica ficam com valor ausente e flag de cobertura.
- **Pre-requisito:** rubrica Gemini em pelo menos 90% das linhas de treino.
- **Regra:** CB_FINAL vira o modelo final se o MAE macro melhorar com p < 0,05 (permutacao pareada por canal, uma comparacao, sem correcao). Caso contrario, o modelo final e o CB_EMB.
- **Descritivo (nao decide):** contribuicao de cada bloco por ablacao (CB_EMB + KNN, CB_EMB + Gemini) e SHAP do CB_FINAL; isso substitui o E-09 como analise de ganho multimodal.
- **Depois da decisao:** treino final em todo o treino, calibracao conformal do intervalo com os residuos do rolling-origin do modelo escolhido, e **teste final aberto uma unica vez**.

### Resultado do E-12 (selecao de modelo)

Notebook `notebooks/10_e12_model_selection.ipynb` (Colab, CPU); artefatos em `artifacts/feature_experiments/e12_model_selection/`.

| Braco | MAE | Pares (justo) | Spearman | vs CB_EMB (metrica da regra) | p | p Holm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| CB_KNN (+ comparaveis) | **0,8625** | 0,589 | 0,230 | MAE -0,020 (17/26 canais) | 0,025 | 0,148 |
| CB_W (peso igual por canal) | 0,8805 | **0,593** | **0,246** | MAE -0,002 | 0,81 | 1,0 |
| **CB_EMB (referencia)** | 0,8825 | 0,581 | 0,235 | - | - | - |
| CB_RANK (YetiRank) | 0,8837 | 0,582 | 0,229 | pares +0,001 | 0,96 | 1,0 |
| ENS (CB_EMB + LAD) | 0,8846 | 0,567 | 0,193 | MAE +0,002 | 0,82 | 1,0 |
| LGBM | 0,8890 | 0,576 | 0,219 | MAE +0,007 | 0,58 | 1,0 |
| XGB | 0,8907 | 0,556 | 0,166 | MAE +0,008 | 0,38 | 1,0 |
| ET | 0,9235 | 0,569 | 0,201 | MAE +0,041 (pior) | 0,002 | 0,012 |
| B0 | 0,9338 | 0,402 | -0,263 | - | - | - |

- **Regra pre-registrada: nenhum candidato aprovado; fica o CB_EMB.** O CatBoost e a escolha certa de algoritmo; LightGBM/XGBoost empatam ou pioram e ExtraTrees e significativamente pior.
- **CB_KNN** (desvio medio dos 5 Shorts anteriores do canal com thumbnail mais parecida) e o unico com sinal (p bruto 0,025), mas nao sobrevive a correcao de Holm. Segue como candidato no E-13 (comparacao unica).
- **Objetivo de ranking (YetiRank)** nao melhora a ordenacao.
- **Optuna (40 tentativas, TPE)**: melhor MAE 0,858 nas origens 1-2 usadas na busca, mas **pior na origem 3** nao usada (0,8985 vs 0,8947; +0,004, p = 0,76, 14/26 canais). Ganho na busca que nao se repete fora dela: sobreajuste a validacao. **Ficam os hiperparametros do E-08.**
- **PyCaret**: a instalacao no Colab (Python 3.13) estourou o limite de 10 min; nao executado (era exploratorio, nao decide).

## Protocolo de experimento

1. Criar um grupo pequeno e coerente de features, sem alterar o teste final.
2. Rodar B0, M1, M2 e M3 no rolling-origin atual, com e sem o grupo novo.
3. Comparar MAE macro por canal, RMSE, vies e ranking por pares em cada origem.
4. Registrar bootstrap e permutacao das diferencas; observar tambem o erro por canal e por tipo de canal.
5. Marcar o grupo como `retida` apenas se o ganho for consistente entre cortes e nao piorar materialmente um grupo.
6. Depois de selecionar a configuracao, rodar uma unica vez no teste final e registrar o resultado.

## Diario de experimentos

| Experimento | Features adicionadas/removidas | Rolling-origin | Teste final | Status | Decisao |
| --- | --- | --- | --- | --- | --- |
| E-00 | Features atuais do baseline | Executado: arquitetura instavel entre M1, M2 e M3 | Executado como referencia inicial | `baseline` | Nenhuma arquitetura vencedora ainda. |
| E-01 | Lote T-01 a T-11: texto e metadados | Piorou MAE medio em M1, M2 e M3 | Melhora pontual, mas nao usada para decisao | `sem_ganho` | Nao reter o lote; testar subgrupos menores depois. |
| E-02 | `has_cta`, `title_tag_overlap`, `duration_bucket` | Delta M2 de -0,0115; IC 95% inclui zero; p=0,3784 | Melhora pontual, nao usada para decisao | `sem_ganho` | Nao reter sem mais evidencia; manter como candidatas. |
| E-03 | Historico: tendencia, volatilidade, views, comentarios e taxa de likes | Piorou rolling-origin para M1, M2 e M3; B0 foi o menor erro | Melhora do teste final nao usada para decisao | `sem_ganho` | Nao reter o lote; revisar transformacoes em um experimento menor. |
| E-04 | Mediana dos ultimos 10 posts e volatilidade dos ultimos 5 | Melhora pequena em M1, mas M2 e M3 nao superam E-00 | Referencia; nao escolher pelo teste final | `sem_ganho` | Nao reter; priorizar nova fonte de informacao em vez de mais agregados historicos. |
| E-05 | TF-IDF de titulo, descricao e tags, combinado ao baseline | M1 caiu de 0,7798 para 0,7535; melhorou em 9 de 12 canais, mas IC 95% do delta E-05 - E-00 foi [-0,0646; 0,0012] e p=0,1162 | Nao executado: teste final segue fechado | `sem_ganho` | Sinal promissor, mas ainda insuficiente para reter; investigar representacao semantica mais robusta ou ampliar a amostra. |
| E-06 | Thumbnail: `thumbnail_v1` + flags de `thumbnail_ocr_v1` (so automotivo, 14 canais, 1.124 linhas) | M1 -0,031 (IC 95% [-0,074; 0,010], p=0,19, 10 de 14 canais melhoram, melhora nas 3 origens); M3 -0,040 (p=0,11); M2 -0,008 (p=0,63). Nenhum modelo supera B0 | Nao executado: teste final segue fechado | `sem_ganho` | Direcao consistente mas nao significativa; manter como candidatas fortes e reavaliar com moda incluida e mais canais. |
| E-07 | Nova formulacao: prever o desvio em relacao a mediana do canal (B0) com perda absoluta (LAD) | Todos os setores: R_M2 0,9275 vs B0 0,9346 e vs M2 0,9794 (delta vs M2 -0,052, p=0,018; 21 de 26 canais melhoram). Empata com B0 (delta -0,007, p=0,45), mas vence B0 nas 3 origens. Thumbnail em cima da nova formulacao: R_M2 -0,019 (p=0,22), M1/M3 ~0 | Nao executado: teste final segue fechado | `retida` (formulacao) / `sem_ganho` (thumbnail) | Adotar residual + LAD como nova referencia; proximas features devem ser comparadas contra R_M2, nao contra M2. |
| E-08 | CatBoost residual (perda MAE) + embedding SigLIP 2 da thumbnail; bracos BASE, THUMB, EMB, FULL | CB_EMB 0,8815 vs B0 0,9338 (delta -0,052, p=0,0006, 18/26 canais) e vs R_M2 (p=0,0002); vence B0 nas 3 origens; pares justo 0,57; controle de idade confirma sinal de conteudo | Nao executado: teste final segue fechado | `retida` (embedding) / `sem_ganho` (thumbnail manual) | CB_EMB vira referencia; calibrar intervalo (cobertura 50-53% para meta de 80%). |
| E-10a/c | PCA 8/32 e CB_FULL com parametros fixos | Nenhum supera CB_EMB_pca16 (p >= 0,50) | Nao executado | `sem_ganho` | Mantem CB_EMB PCA 16 (regra pre-registrada). |
| E-11 | Canal nao visto (leave-one-channel-out dentro do rolling-origin) | CB_EMB_loco: MAE 0,951 vs B0 0,934 (p=0,36); pares 0,553 > 0,5 (p=0,002, 22/26 canais); CB_BASE_loco no acaso | Nao executado | diagnostico | Generaliza para ordenar criativos de canal novo; nivel absoluto nao. |
| E-10b | Embedding de texto (mpnet multilingue) sobre o CB_EMB | Titulo: MAE -0,004 (p=0,73); titulo+descricao: -0,006 (p=0,58) | Nao executado | `sem_ganho` | Texto semantico nao acrescenta ao embedding da thumbnail. |

| E-12 | LGBM, XGB, ET, ensemble, YetiRank, comparaveis, peso por canal; Optuna; PyCaret | Nenhum aprovado apos Holm; CB_KNN MAE -0,020 (p bruto 0,025, Holm 0,148); Optuna pior na origem 3 (p=0,76); PyCaret nao instalou | Nao executado | `sem_ganho` (algoritmos, Optuna) / candidato (comparaveis) | Mantem CB_EMB com parametros do E-08; comparaveis vao ao E-13. |
