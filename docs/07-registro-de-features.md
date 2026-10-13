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
