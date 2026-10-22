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


### D-015 - Destilacao para modelos por canal (professor grande, alunos por canal)

- **Data**: 12/10/2026
- **Tipo**: ideia de arquitetura
- **Status**: decidida (usar o modelo grande como extrator de features; destilacao de rotulos descartada)
- **O que foi proposto**: um modelo grande ("professor") analisa o YouTube como um todo e gera previsoes de engajamento; essas previsoes, separadas por canal ou nicho, viram o alvo de treino de modelos pequenos por canal ("alunos", ex.: XGBoost), que rodariam barato no dia a dia.
- **Pontos de atencao**:
  1. Temos o rotulo real (`like_count`) para todo Short. Treinar o aluno na previsao do professor limita o aluno a qualidade do professor e herda seus vieses; destilacao so compensa quando falta rotulo ou o professor e comprovadamente melhor.
  2. O TAP exige modelo treinado sobre dados observados e diz que o LLM "nunca chuta o numero"; usar a previsao do LLM como alvo e uma forma indireta de fazer isso.
  3. Modelos por canal tem poucos dados (~92 Shorts de treino por canal em media, alguns com ~40); ver D-004.
  4. Retencao, CTR e dados globais do YouTube nao estao disponiveis pela API publica (D-001).
- **O que ja aproveitamos da ideia**:
  - Modelo grande como professor de **atributos**, nao de rotulos: o Gemini assiste ao video e devolve a rubrica do TAP (`scripts/coleta/collect_gemini_video_features.py`); o CatBoost aprende essas colunas contra as curtidas reais. Custo do modelo grande pago uma vez por video; previsao barata depois.
  - Personalizacao por canal via alvo residual (D-014): o B0 de cada canal se atualiza a cada post e o modelo global aprende so o desvio.
  - Especializacao no nivel com dados suficientes: tipo de canal (M2) e setor (D-004). Por canal, a alternativa segura e pooling parcial (efeito do canal encolhido para a media global quando ha pouco historico).
- **Teste que decide**: o piloto Gemini coleta `potencial_viral_0a10`, que e a "previsao do professor". Medir Spearman desse campo com `target_residual_log_likes` (dentro do canal). Perto de zero: o professor nao tem conhecimento a destilar. Com sinal: entra como feature no CatBoost (E-09). Cobre tambem a extensao opcional do TAP "Comparacao LLM zero-shot vs. modelo treinado".
- **Resultado do teste (12/10/2026)**: piloto de 100 Shorts (`data/bronze/gemini_video/20261013T005518Z`, `gemini-3.5-flash-lite`, 100/100 ok, ~5,5 mil tokens de entrada por video). Somente linhas de treino (83 videos, 25 canais; as 19 de teste nao foram olhadas). `potencial_viral_0a10` tem Spearman 0,32 com `target_residual_log_likes` e 0,21 com o residuo centrado no canal (limiar de p<0,05 com n=83 e ~0,22). Outros campos com sinal parecido: `texto_na_tela` (0,28), `brilho_saturacao_0a10` (0,25), `forca_hook_0a10` (0,22). Ha 16 campos testados, entao isso e indicio, nao confirmacao.
- **Conclusao**: o professor tem algum sinal, mas fraco e ruidoso, e as notas se concentram em 6-8. Confirma o uso correto: rubrica do Gemini como **features** do CatBoost treinado em curtidas reais (E-09), nao como alvo de treino de modelos por canal. A destilacao como proposta segue nao recomendada.
- **Proximo passo**: coletar a rubrica para todos os Shorts do Gold e medir o ganho no E-09 contra o CB_EMB.

### D-016 - Embedding visual da thumbnail supera o B0; CB_EMB vira referencia

- **Data**: 12/10/2026
- **Tipo**: resultado de experimento
- **Status**: retida
- **O que descobrimos**: no E-08, CatBoost no alvo residual com embedding SigLIP 2 da thumbnail (PCA 16) teve MAE 0,8815 contra 0,9338 do B0 (delta -0,052, IC 95% [-0,081; -0,025], p=0,0006, 18 de 26 canais) e venceu nas 3 origens. Sem o embedding, o CatBoost empata com o R_M2: o ganho vem do conteudo visual, nao do algoritmo. Ranking justo por pares 0,57-0,58 e Spearman intra-canal ~0,2. Controle por idade do video confirma sinal de conteudo.
- **Decisao**: CB_EMB passa a ser a referencia para os proximos experimentos. Features manuais de thumbnail continuam `sem_ganho`.
- **Pendencias**: intervalo de 80% cobre so 50-53% (calibrar com conformal); somar rubrica Gemini (E-09); abrir o teste final uma unica vez depois de congelar a configuracao. Detalhes em `docs/07` (E-08).

### D-017 - Em canal nunca visto, o modelo ordena criativos mas nao acerta o nivel

- **Data**: 12/10/2026
- **Tipo**: resultado de experimento
- **Status**: decidida
- **O que descobrimos**: no E-11 (canal fora do treino), o CB_EMB acerta 55,3% dos pares dentro do canal (p=0,002, 22 de 26 canais acima do acaso), enquanto o modelo sem embedding fica no acaso. O MAE, porem, nao supera a mediana do canal (B0). No E-10, PCA 8/32, a combinacao CB_FULL e o embedding de texto (titulo ou titulo + descricao) nao superam o CB_EMB com PCA 16 (todos p >= 0,46).
- **Decisao**: CB_EMB (PCA 16) segue como referencia. Para clientes novos, comunicar a tool como ferramenta de **ordenacao** de criativos do mesmo canal; a previsao absoluta vem com confianca baixa (ja implementado). Base para a resposta a "funciona para clientes novos?" na banca.

### D-018 - Trocar de algoritmo ou ajustar mais hiperparametros nao melhora o modelo

- **Data**: 13/10/2026
- **Tipo**: resultado de experimento
- **Status**: decidida
- **O que descobrimos**: no E-12, LightGBM, XGBoost, ExtraTrees, ensemble, objetivo de ranking e peso por canal nao superam o CB_EMB (Holm). O Optuna melhorou as origens usadas na busca e piorou na origem 3 nao usada (sobreajuste a validacao). A unica ideia com sinal e a feature de comparaveis (p bruto 0,025, nao aprovada apos Holm).
- **Decisao**: CatBoost com os hiperparametros do E-08 e o algoritmo final. Melhorias restantes vem de dados (comparaveis, rubrica Gemini), decididas uma unica vez no E-13 (protocolo em `docs/07`).

### D-019 - Modelo final: embedding + comparaveis + rubrica Gemini (E-13), confirmado no teste

- **Data**: 13/10/2026
- **Tipo**: resultado de experimento
- **Status**: decidida
- **O que descobrimos**: com a rubrica Gemini coletada para 99,97% dos Shorts, o CB_FINAL (CB_EMB + comparaveis visuais + rubrica + fala) reduz o MAE do rolling-origin de 0,882 para 0,850 (p = 0,003, 21 de 26 canais). Sozinhos, comparaveis e rubrica nao sao significativos; juntos sao. No teste final, aberto uma unica vez, o CB_FINAL tem MAE 0,746 (B0 0,858; CB_EMB 0,780), acerta 58% dos pares do mesmo canal (B0 44%) e o intervalo de 80% cobre 79,9%.
- **Decisao**: CB_FINAL e o modelo final da tool. O teste foi usado e nao deve orientar novas escolhas; melhorias futuras precisam de dados novos (mais canais, horizonte fixo) para nova avaliacao.


### D-020 - O coletor antigo cortava canais pelo teto de 300 uploads

- **Data**: 20/10/2026
- **Tipo**: qualidade de dados
- **Status**: resolvido
- **O que descobrimos**: `collect_data_discovery.py` lia so os 300 uploads mais recentes de cada canal. Em canais que postam muito, a janela de 18 meses ficava truncada: no Gold v1, ComprecarTV tinha so 2 semanas de Shorts (75), Giuliana Mafra 2 meses, Acelerados e Edi Ribeiro 4 meses. Sem o teto, ComprecarTV tem 992 Shorts elegiveis e Acelerados 566.
- **Decisao**: novo coletor so com biblioteca padrao (`scripts/coleta/collect_youtube_stdlib.py`, roda no Windows do Inteli), que pagina a playlist de uploads ate sair da janela, aceita `@handle` e retoma por canal. Resolvedor de canais por handle com busca de reserva e triagem de Shorts (`resolve_channels_stdlib.py`).

### D-021 - Base v2: canais equilibrados entre pessoa, marca e midia, ate 150 Shorts por canal

- **Data**: 20/10/2026
- **Tipo**: decisao de dados
- **Status**: decidida
- **O que foi feito**: busca de candidatos nos rankings da HypeAuditor (YouTube Autos & Vehicles e Fashion, Brasil) e em listas de marcas e veiculos. Coleta de 233 canais (119 automotivo, 114 moda): 91 e 76 com >= 30 Shorts, 44 mil Shorts elegiveis. Revisao dos titulos e exclusao de canais fora do tema (`data/candidates/exclusions_v2.csv`).
- **Decisao**: em vez de usar todos os canais, equilibrar por grupo: automotivo 13 marcas, 13 pessoas e 11 midias; moda 13 marcas, 13 pessoas e 5 midias (todas as midias de moda com Shorts suficientes). Ate 150 Shorts mais recentes por canal no treino; o historico do canal (B0) usa todos os Shorts coletados. Resultado: Gold v2 com 8.398 Shorts e 68 canais (antes 3.004 e 26).
- **Por que**: sem teto, quatro canais (Mobiauto, Star Motors, AutoPapo, Doutor Multas) teriam 900-1.400 Shorts cada e dominariam o treino; o teto tambem limita o custo do Gemini (~6,9 mil Shorts novos).
- **Beleza**: fora do recorte, por decisao do time (dois setores, como permitido pelo TAP).

### D-022 - Sete criadores de moda do v1 nao eram de moda

- **Data**: 20/10/2026
- **Tipo**: qualidade de dados
- **Status**: corrigido no v2
- **O que descobrimos**: lendo os titulos, Giuliana Mafra, Edi Ribeiro e Matando Matheus a Grito sao humor; Mandy Candy e vlog; Vique e jogo (Dress to Impress); Ingryd Lourenco e costura; NiinaSecrets e maquiagem. No v1, "moda / criador" era na pratica humor e vlog. Marcas de moda do v1 (Renner, C&A, Hering, Maria Bardela) estao corretas.
- **Decisao**: os sete ficam fora do Gold v2; os resultados do v1 continuam validos como estao, mas o recorte "moda criador" do v1 deve ser lido com esse aviso.

### D-023 - Camada multimodal: Gemini pela URL publica em vez de baixar video

- **Data**: 20/10/2026
- **Tipo**: decisao de escopo
- **Status**: decidida
- **Contexto**: a camada minima do TAP sugere frames, Whisper, OCR, rosto, YOLO, OpenCV e PySceneDetect sobre o video baixado. Download de Shorts e bloqueado por IP na rede do Inteli e no celular (D-011) e esbarra nos Termos do YouTube.
- **Decisao**: nao baixar video. O Gemini assiste ao Short pela URL publica e preenche a rubrica do TAP, que cobre os mesmos atributos (gancho, ritmo de cortes, musica, fala, rosto, texto na tela, produto, estetica); a thumbnail entra via SigLIP 2. O ganho foi medido no E-13 (p = 0,003). Candidatos sem download (embedding semantico da fala, snapshots de metricas para alvo D7/D30) estao em `docs/11` e na branch `feat/multimodal`, e so entram no modelo com ablacao temporal.

### D-024 - Extensoes da secao 4.2 do TAP implementadas

- **Data**: 20/10/2026
- **Tipo**: entrega
- **Status**: implementadas
- **LLM zero-shot vs modelo treinado** (`scripts/experimentos/run_llm_zero_shot.py`): no teste final do E-13, a nota `potencial_viral_0a10` do Gemini sozinha acerta 53,9% dos pares do mesmo canal (p = 0,08 contra o acaso; 19 de 26 canais acima de 50%); o CB_FINAL acerta 58,0% (+4,0 p.p., p = 0,15, 17 de 26 canais). O LLM tem sinal de ordenacao, mas nao da numero nem intervalo; a nota concentra 82% dos Shorts entre 6 e 8. **Atualizacao 21/10/2026 (Gold v2.1):** no teste do E-15 (1.236 Shorts, 68 canais) o LLM acerta 56,2% dos pares (p < 0,001 vs acaso, 49/68) e o modelo treinado 62,4%: +6,2 p.p. (IC95% [3,8; 8,5], p < 0,001, 48/68). No E-14 (modelo do v1 congelado, 4.884 Shorts) a diferenca e +3,9 p.p. (p < 0,001). Com mais canais, a vantagem do modelo treinado passa a ser significativa.
- **What-if** (`Predictor.what_if`, tool MCP `simular_variacoes`): re-pontua o criativo mudando um fator por vez (horario, dia, duracao, titulo, criativo alternativo). O modelo ve o titulo so pelo tamanho; a saida avisa que diferencas menores que o intervalo nao sao decisivas.
- **Treino dinamico** (`galaxies_agent.trainer`, tool MCP `treinar_modelo_cliente`): base do cliente no esquema minimo do TAP -> features so com posts anteriores, CatBoost residual, avaliacao temporal contra a mediana do canal, intervalo calibrado e decisao (usar / usar com cautela / nao usar). Exemplo com as 16 marcas do v1, so metadados: pares +10 p.p. (p = 0,017), MAE sem diferenca significativa -> "usar com cautela, para ordenar".

### D-025 - Gold v2.1: 100 Shorts por canal para fechar a rubrica do Gemini

- **Data**: 21/10/2026
- **Tipo**: decisao de dados
- **Status**: decidida
- **Contexto**: o credito pre-pago do Gemini acabou duas vezes (HTTP 402); na madrugada de 21/10 sairam 2.429 rubricas, deixando o Gold v2 (150 por canal) com 47,6% de cobertura e 43 canais abaixo de 50%.
- **Decisao**: manter os 68 canais e reduzir para 100 Shorts por canal, priorizando os que ja tem rubrica (amostra aleatoria do canal, porque o coletor sorteia a ordem) e completando com os mais recentes. Faltam 2.547 rubricas (~50 h de video, estimativa ~US$ 4,50), rodadas em dois lotes: A = teste (619) e B = treino (1.929). Cortar canais foi descartado: economizaria pouco (os canais pequenos ja estavam cobertos) e reduziria o poder das comparacoes por canal.

### D-026 - Tool passa a servir o modelo treinado no Gold v2.1

- **Data**: 21/10/2026
- **Tipo**: decisao de modelo
- **Status**: implementada
- **Contexto**: a coleta da rubrica fechou (Gold v2.1 com 99,97% de rubrica; 2 Shorts barrados pelo filtro de conteudo do Gemini) e o embedding v2 cobriu 100%. E-14 e E-15 rodaram uma vez cada (docs/07).
- **Decisao**: pela regra pre-registrada do E-15, o pacote `tool_v3_20261021T175806Z` (todo o Gold v2.1, mesma arquitetura e hiperparametros) substitui o `tool_v2_20261014T000408Z` em `artifacts/tool_model/LATEST.json`. O pacote antigo continua no repositorio e e reproduzivel com `train_tool_model.py`.
- **Intervalo**: o E-14 mostrou que a calibracao do v1 cobre 72,8% nos dados novos; o `tool_v3` recalibra com rolling-origin no Gold v2.1 (80,8% de cobertura na origem 3).
- **Canais fora do v2**: os 12 canais da Silver v1 que nao estao no Gold v2 (os sete criadores de D-022 e cinco canais que nao entraram na selecao do v2: Orlivya, Riachuelo, Isa Moda Brasil, FARM Rio e FlatOut Brasil) continuam no historico da tool, com `in_training = False`: a tool preve pelo `channel_id` e marca confianca baixa.
- **Embedding**: o cache da tool passa a ser `thumbnail_siglip2_v1_v2/20261021T172500Z/embeddings.npz` (v1 + v2, conjuntos disjuntos, mesmo modelo e preprocess).
- **Testes**: os testes da tool leem o Gold e as Silvers do pacote servido (`metadata.json`), nao o LATEST dos experimentos; 24 testes passam com o pacote novo e com o antigo.

### D-027 - Inscritos: no contrato e no baseline do TAP, fora do modelo

- **Data**: 21/10/2026
- **Tipo**: decisao de modelo / contrato
- **Status**: implementada
- **Contexto**: o TAP cita inscritos na coleta, no baseline ("regressao linear sobre inscritos") e no contrato (`channel_stats`: "inscritos e historico do canal"). O coletor ja gravava `subscriber_count_at_collection` no Bronze, mas o dado nao chegava a Silver nem a tool.
- **O que foi feito**: Silver `channel_stats` (`build_channel_stats.py`, 236 canais, sem chamar a API); baseline do TAP no teste do E-15 (`run_subscriber_baseline.py`): MAE macro 1,55 (so inscritos) e 1,26 (+ setor e tipo) contra 0,91 da mediana do canal e 0,74 do modelo final; campo opcional `ChannelStats.subscribers` e saida `channel_subscribers`.
- **Decisao**: inscritos nao entram como feature. Sao uma fotografia da data da coleta (o canal cresceu depois de cada post, em parte por causa dele), sao constantes dentro do canal (nao ajudam a escolher entre criativos do mesmo canal) e, como baseline, erram bem mais que a mediana do canal. Ficam no contrato como informacao.

