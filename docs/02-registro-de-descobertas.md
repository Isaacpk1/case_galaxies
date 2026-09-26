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
- **Proximo passo**: executar validacao rolling-origin e incluir atributos de conteudo antes de tomar uma decisao de produto.
