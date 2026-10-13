# Registro de Descobertas e Ideias

Este documento e um backlog vivo de aprendizados, hipoteses e possibilidades encontradas durante o projeto. O objetivo e nao perder uma ideia antes de ela virar decisao, experimento ou item de roadmap.

## Como registrar uma descoberta

Para cada item novo, anotar:

- **Data**: quando a ideia apareceu.
- **Tipo**: descoberta, hipotese, oportunidade, risco ou decisao.
- **Status**: aberta, para experimentar, decidida, roadmap ou descartada.
- **O que descobrimos**: descricao curta e objetiva.
- **Impacto**: por que isso importa para o projeto.
- **Proximo passo**: acao concreta, se houver.

## Itens registrados

### D-001 - MCP para dados autorizados de parceiros

- **Data**: 24/09/2026
- **Tipo**: oportunidade de integracao
- **Status**: roadmap
- **O que descobrimos**: a YouTube Data API publica oferece visualizacoes, curtidas e comentarios, mas nao oferece impressoes, CTR de thumbnail, retencao, tempo assistido ou origem de trafego. Com autorizacao do parceiro, um servidor MCP da Galaxies pode consultar a YouTube Analytics API para os canais que pertencem ao parceiro.
- **Impacto**: o modelo pode receber historicos mais ricos e a analise pode explicar nao apenas se um video teve views, mas como ele foi distribuido e retido pela audiencia.
- **Regra de vazamento**: metricas do proprio video apos a publicacao continuam sendo resultados, nunca entradas para prever esse mesmo video. Apenas agregados de posts anteriores do canal podem entrar como features.
- **Proximo passo**: definir, no futuro, o contrato de entrada do MCP para dados autorizados e os campos minimos de Analytics.

### D-002 - Separar marcas e criadores

- **Data**: 24/09/2026
- **Tipo**: descoberta de dados
- **Status**: decidida
- **O que descobrimos**: canais de marcas e criadores possuem volumes e distribuicoes de curtidas diferentes.
- **Impacto**: o campo `channel_type` deve permanecer na base para o modelo nao atribuir ao conteudo um efeito que vem do tipo e tamanho da audiencia do canal.
- **Proximo passo**: usar `channel_type` nas analises exploratorias e avaliar seu efeito no modelo de baseline.

### D-003 - Curtidas relativas ao historico do canal

- **Data**: 24/09/2026
- **Tipo**: hipotese de modelagem
- **Status**: para experimentar
- **O que descobrimos**: comparar curtidas absolutas entre canais grandes e pequenos nao mede apenas a qualidade do criativo.
- **Impacto**: alem de prever `log(1 + curtidas)`, devemos testar uma medida relativa, como curtidas do video divididas pela mediana dos posts anteriores do mesmo canal.
- **Proximo passo**: comparar os dois alvos no experimento de baseline com validacao temporal.

### D-004 - Modelo central multicanal; especializacao por parceiro nao e premissa

- **Data**: 24/09/2026
- **Tipo**: decisao de modelagem
- **Status**: decidida
- **O que descobrimos**: o TAP define como nucleo um modelo preditivo tabular sobre atributos do criativo, do canal e do contexto, com baseline de media ou mediana historica do canal. A entrega espera uma base curada multicanal. O retreino de um modelo especialista com uma base liberada por cliente aparece apenas como extensao opcional.
- **Decisao**: o piloto treinara um modelo global multicanal, usando o contexto e o historico anterior de cada canal. Nao vamos depender da hipotese de que um parceiro entregara centenas ou milhares de posts proprios.
- **Impacto**: o baseline continua sendo calculado por canal, mas o modelo aprende com todos os canais do piloto. Isso evita tentar treinar modelos separados com poucas dezenas de exemplos por canal.
- **Proximo passo**: comparar, na fase de modelagem, um modelo global com a feature `sector` contra dois modelos globais separados por setor. Modelos por canal ficam fora do escopo do piloto.

### D-005 - Hipoteses de segmentacao por tipo de canal

- **Data**: 25/09/2026
- **Tipo**: hipotese de modelagem
- **Status**: para experimentar
- **O que descobrimos**: marcas e criadores possuem escalas e dinamicas diferentes de publicacao, mas o recorte atual ainda oferece poucas observacoes para modelos por canal.
- **Decisao experimental**: comparar um modelo global unico com `channel_type` contra dois modelos globais especializados, um para marcas e outro para criadores. Ambos serao comparados ao baseline de mediana historica por canal.
- **Criterio**: decidir pelo MAE medio por canal no teste temporal; nao pelo erro agregado que poderia ser dominado pelos canais com mais Shorts.
- **Proximo passo**: implementar B0, M1, M2 e M3 conforme [Hipoteses de modelagem](05-hipoteses-de-modelagem.md).

### D-006 - Primeiro baseline favorece o modelo global, sem evidencia conclusiva

- **Data**: 25/09/2026
- **Tipo**: resultado de experimento
- **Status**: para confirmar em novos cortes temporais
- **O que descobrimos**: no teste temporal de 375 Shorts, M1 (modelo global com `channel_type`) teve MAE macro de 0,7729 em `log(1 + curtidas)`, contra 0,7929 do baseline historico B0. M2 (modelos separados para marcas e criadores) teve 0,7863 e M3 (global sem `channel_type`) teve 0,7763.
- **Evidencia estatistica**: a diferenca M2 - M1 foi 0,0134, com IC bootstrap de 95% [-0,0198, 0,0443] e p-valor de permutacao 0,4468. Portanto, nao ha evidencia de que separar marcas e criadores melhore a previsao neste recorte. A diferenca M1 - B0 tambem e inconclusiva com apenas 12 canais.
- **Decisao provisoria**: manter M1 como referencia pela menor MAE observada e menor complexidade. Nao concluir que ele e superior ate repetir o teste em mais cortes temporais ou aumentar o numero de canais.
- **Proximo passo**: incluir atributos de conteudo antes de tomar uma decisao de produto.

### D-007 - Rolling-origin confirma instabilidade entre arquiteturas

- **Data**: 26/09/2026
- **Tipo**: resultado de experimento
- **Status**: para confirmar com novas features e mais canais
- **O que descobrimos**: em tres janelas expansivas dentro do treino, M2 teve MAE macro medio de 0,7615, M3 de 0,7620, B0 de 0,7734 e M1 de 0,7798. O vencedor mudou entre os cortes: M1/M3 foram melhores no primeiro, M1 no segundo e M2 no terceiro.
- **Evidencia estatistica**: M2 - M1 teve diferenca de -0,0183, mas o IC bootstrap de 95% [-0,0552, 0,0177] inclui zero e o p-valor de permutacao foi 0,3662. Portanto, separar marcas e criadores ainda nao tem evidencia conclusiva.
- **Impacto**: a arquitetura nao deve ser escolhida pelo unico teste final. O baseline historico e os modelos devem continuar como comparadores enquanto adicionamos atributos de conteudo.
- **Proximo passo**: criar features textuais e de metadados, repetir rolling-origin e depois consultar o teste final apenas para a configuracao escolhida.

### D-008 - Registro de features passa a guiar os experimentos

- **Data**: 26/09/2026
- **Tipo**: decisao de processo
- **Status**: decidida
- **Decisao**: toda feature candidata sera registrada com fonte, definicao, regra antileakage, status e resultado antes de ser incorporada ao modelo.
- **Impacto**: evita adicionar variaveis sem hipotese e preserva rastreabilidade entre o que foi testado, o que funcionou e o que foi descartado.
- **Referencia**: [Registro de features e experimentos](07-registro-de-features.md).

### D-009 - Ablation reduzida ainda nao confirma ganho de features textuais

- **Data**: 26/09/2026
- **Tipo**: resultado de experimento
- **Status**: inconclusivo
- **O que descobrimos**: a ablation de CTA, sobreposicao titulo/tags e faixa de duracao reduziu o MAE medio de M2 no rolling-origin, mas a diferenca pareada por canal teve IC de 95% incluindo zero e p-valor de permutacao de 0,3784.
- **Decisao**: nao reter o lote para produto. As variaveis permanecem candidatas para reavaliacao junto de features historicas ou com mais canais.

### D-010 - Lote amplo de historico tambem nao generalizou

- **Data**: 26/09/2026
- **Tipo**: resultado de experimento
- **Status**: sem ganho
- **O que descobrimos**: medias, volatilidade, tendencias, views, comentarios e taxa historica de likes reduziram MAE no teste final, mas pioraram o rolling-origin em relacao ao baseline; B0 teve o menor MAE medio nessa validacao.
- **Decisao**: nao reter o lote historico. Revisar apenas subgrupos pequenos e transformacoes robustas em um experimento futuro.

### D-011 - Coleta de legendas bloqueada por IP; Apify em teste

- **Data**: 30/09/2026
- **Tipo**: descoberta operacional e decisao
- **Status**: em teste
- **O que descobrimos**: o run `20260930T025936Z` tentou 1.506 Shorts sem pausa entre videos e obteve 0 transcricoes (913 `IpBlocked`, 349 `RequestBlocked`, 215 `TranscriptsDisabled`, 27 `NoTranscriptFound`, 2 `VideoUnplayable`). O mesmo video baixado em 29/09 passou a falhar, confirmando bloqueio por IP. Na rede do Inteli e no roteador do celular, `youtube-transcript-api` e `yt-dlp` (com Deno, curl-cffi e PO Token via bgutil) receberam `IpBlocked`/HTTP 429 mesmo pedindo uma unica faixa de legenda.
- **Evidencia**: runs `20260930T165151Z` a `20260930T171743Z` e teste manual do `yt-dlp -v` com `pot=` na URL do `timedtext` retornando 429.
- **Decisao**: o coletor passou a ter pausa com jitter, circuit breaker, retomada, uma faixa por video e o provider `apify` para testar a coleta a partir de infraestrutura de terceiro. Os 215 `TranscriptsDisabled` obtidos durante o bloqueio podem ser falsos negativos e devem ser rechecados.
- **Teste Apify (run `20260930T173000Z`)**: 5/5 transcricoes, mas 3 vieram em arabe porque o actor `karamelo/youtube-transcripts` retorna a primeira faixa de videos com dublagem automatica. O padrao passou a ser `dltik/youtube-transcript-scraper`, que aceita `preferredLanguages`.
- **Impacto**: o experimento de texto (E-02/E-05) nao depende de legendas e segue com titulo, descricao e tags. A transcricao (E-04) passa a depender do resultado do teste com Apify ou de uma rede residencial.

### D-012 - Legendas do actor `karamelo` vieram traduzidas para o ingles

- **Data**: 01/10/2026
- **Tipo**: qualidade de dados
- **Status**: mitigado na Feature Store; correcao pendente na coleta
- **O que descobrimos**: das 323 legendas validas, a deteccao de idioma por stopwords classificou 165 como portugues, 118 como ingles, 3 como frances/espanhol e 37 como indefinidas (quase so `[music]`). Todas as 118 em ingles vieram de `karamelo/youtube-transcripts`; o texto e uma traducao automatica da fala em portugues (ex.: todos os 18 Shorts do ComprecarTV). O manifesto nao trazia `language_code` para esses casos, entao a validacao nao pegou.
- **Decisao**: `transcript_v1` mantem esses Shorts com `transcript_is_auto_translation = 1`. Contagens e tempos (palavras, densidade, gancho de 5s) continuam validos porque a traducao preserva os timestamps; o CTA falado tambem procura termos em ingles. Para features semanticas (embedding da fala), usar so `transcript_is_portuguese = 1` ate recoletar esses videos com um actor que respeite `preferredLanguages`.

### D-013 - Thumbnails de Shorts sao um frame vertical com laterais desfocadas

- **Data**: 01/10/2026
- **Tipo**: qualidade de dados
- **Status**: resolvido
- **O que descobrimos**: cerca de 62% das thumbnails de automotivo (96% nos criadores) sao o frame 9:16 no centro de uma imagem 16:9, com copias desfocadas e escurecidas nas laterais. Medir a imagem inteira puxava o brilho medio de 0,37 para 0,26 e fazia o OCR ler o texto duas vezes.
- **Decisao**: `thumbnail_v1` e o OCR recortam o painel central antes de medir. Detalhes em `docs/07`.

### D-014 - Formulacao residual com perda absoluta vira a referencia

- **Data**: 01/10/2026
- **Tipo**: resultado de experimento
- **Status**: retida
- **O que descobrimos**: o B0 vencia todos os Ridge porque (1) o Ridge prevê o nivel de curtidas do zero e (2) minimiza erro quadratico, prevendo a media, enquanto o MAE premia a mediana. Prevendo o desvio em relacao ao B0 com perda absoluta (E-07), o modelo melhora 0,052 de MAE sobre o M2 antigo (p=0,018, 21 de 26 canais) e empata com o B0.
- **Decisao**: Gold passa a trazer `target_residual_log_likes`. A escolha de algoritmo (CatBoost/LightGBM etc.) continua na fase de modelagem; o Ridge foi usado so como regua de comparacao de features.

