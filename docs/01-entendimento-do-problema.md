# Agente Preditivo de Performance em Midias Sociais

## Status

**Fase primária concluida em 24/09/2026.** Este documento registra as decisoes de entendimento do problema e de viabilidade inicial dos dados. Ele passa a ser a referencia do piloto antes da fase de consolidacao da base e modelagem.

## Problema de negocio

A Galaxies ja utiliza personas sinteticas para emitir opinioes qualitativas sobre criativos. Este projeto adiciona uma camada quantitativa para apoiar uma decisao anterior a publicacao:

> Dado um ou mais criativos ainda nao publicados, qual deles tem maior probabilidade de performar melhor no canal da marca?

O objetivo nao e prometer viralizacao nem substituir testes reais. O objetivo e diminuir a incerteza na priorizacao de criativos antes de investir verba de midia.

## Definicao do MVP

O MVP recebera um YouTube Short, sua legenda, o canal, o setor e o horario planejado de publicacao. Ele retornara uma previsao de curtidas, um intervalo de predicao, um nivel de confianca, a performance relativa ao historico do canal, um ranking quando houver mais de um criativo e uma explicacao baseada em evidencias.

```text
Video + legenda + contexto do canal
              |
              v
Extracao de atributos e consulta ao historico
              |
              v
Modelo preditivo treinado
              |
              v
Previsao + intervalo + ranking + explicacao
```

## Decisoes confirmadas

| Tema | Decisao |
| --- | --- |
| Plataforma do MVP | YouTube Shorts |
| Setores do recorte inicial | Moda e automotivo |
| Metrica primaria | Curtidas |
| Decisao apoiada | Comparar e priorizar criativos antes da publicacao |
| Unidade principal de previsao | Criativo para um canal especifico |
| Historico do canal | Obrigatorio no MVP |
| Papel do modelo | Produzir a previsao estatistica |
| Papel do LLM/agente | Orquestrar ferramentas e explicar evidencias; nunca inventar a previsao |
| Avaliacao | Split temporal e teste em canais nao vistos; split aleatorio nao sera a metrica final |
| Dados estruturados | Abordagem ELT: preservar dados brutos e transformar de modo reproduzivel |
| Midia | ETL cacheado para converter video, audio e imagem em features reutilizaveis |
| Periodo historico do piloto | 18 meses |
| Mercado e idioma | Brasil, com conteudo prioritariamente em portugues |
| Criterio de entrada de canal | Minimo de 30 Shorts elegiveis; preferencia por 50 |
| Canais do piloto | 12 canais: 3 marcas e 3 criadores em cada setor |
| Volume do piloto | 1.902 Shorts elegiveis |
| Estrategia de modelo do piloto | Modelo global multicanal; baseline calculado por canal |

## Casos de uso

### 1. Ranking de criativos - caso principal

O usuario envia de dois a cinco criativos para o mesmo canal. O sistema preve curtidas para cada opcao, ordena os resultados e apresenta as evidencias que sustentam a recomendacao.

Exemplo de resposta esperada:

```text
1. Criativo B: 3.200 curtidas previstas (intervalo: 2.100 a 4.700)
2. Criativo A: 2.500 curtidas previstas
3. Criativo C: 1.800 curtidas previstas
```

### 2. Avaliacao de um criativo unico

O usuario envia um video antes da publicacao e recebe a previsao, o intervalo, a classificacao relativa ao canal e os fatores mais relevantes.

### 3. Baixa evidencia

Quando o historico do canal for insuficiente ou o criativo estiver distante dos dados de treino, a resposta deve ampliar o intervalo e sinalizar baixa confianca. O sistema nao deve apresentar falsa precisao.

## Metrica e alvo preditivo

A metrica primaria do MVP sera `like_count`. Alem do valor absoluto, o sistema deve calcular desempenho relativo ao canal, por exemplo:

```text
curtidas do video / mediana de curtidas dos posts anteriores do canal
```

Esse alvo relativo reduz o efeito do tamanho do canal e torna a comparacao entre criativos mais util.

Metricas futuras - fora do MVP, no roadmap:

- visualizacoes;
- comentarios e compartilhamentos;
- classificacao em faixas de performance;
- indice composto de engajamento;
- objetivos de campanha especificos.

## Hipotese a testar

> A combinacao de historico do canal, contexto de publicacao e atributos de conteudo extraidos do criativo consegue ordenar a performance futura de criativos melhor do que a mediana historica de curtidas do canal.

O baseline minimo sera a media ou mediana historica do canal. O modelo so agrega valor se superar esse baseline em dados futuros.

## Dados e prevencao de vazamento

So podem ser usadas como features informacoes disponiveis antes da publicacao:

- video, thumbnail, frames e audio;
- titulo, descricao e legenda planejada;
- setor e horario planejado;
- inscritos e historico de posts anteriores do canal.

Curtidas, views, comentarios e compartilhamentos do proprio post sao resultados posteriores a publicacao. Eles servem para treinar e avaliar o modelo, mas nao podem ser usados como entrada na previsao daquele post.

## Estrategia de dados proposta

```text
YouTube Data API e datasets publicos
              |
              v
Bronze: JSON e arquivos brutos
              |
              v
Silver: dados limpos, tipados e deduplicados
              |
              v
Gold: features de canal, texto, imagem, video e audio
              |
              v
Treino, avaliacao e registro do modelo
              |
              v
Funcao Python, MCP e agente
```

O processamento multimodal deve ser cacheado por `video_id`: downloads, frames, transcricoes, atributos visuais e embeddings nao devem ser recalculados sem necessidade.

## Limitacao da API publica e evolucao com dados de parceiros

A YouTube Data API publica entrega `view_count`, `like_count` e `comment_count`, mas nao entrega as metricas de distribuicao do video: impressoes da thumbnail, cliques, CTR de thumbnail, retencao, tempo de exibicao e origem do trafego. Em Shorts, `view_count` representa reproducoes e nao deve ser interpretado como numero de cliques na thumbnail.

Se a Galaxies ou um parceiro autorizar acesso aos dados do proprio canal, um servidor MCP podera consultar a YouTube Analytics API e incorporar essas metricas ao mesmo esquema minimo de dados. O acesso deve ocorrer apenas por autenticacao autorizada pelo dono do canal; chaves, tokens e dados de canais externos nao devem ser armazenados no repositorio.

| Dado autorizado do parceiro | Uso permitido no modelo |
| --- | --- |
| Impressoes e CTR de posts anteriores | Feature agregada do historico do canal |
| Retencao e tempo assistido de posts anteriores | Feature agregada do historico e analise de formato |
| Impressoes, CTR e retencao do post previsto | Resultado posterior a publicacao; usar apenas para avaliacao, nunca como feature da previsao daquele post |

Essa integracao melhora a analise do parceiro, mas nao e requisito para o piloto com dados publicos.

## Roadmap: canais sem historico suficiente

O MVP e personalizado para canais com historico. Canais novos ou com poucos posts serao tratados em uma evolucao posterior:

1. Usar padroes agregados do setor e canais similares.
2. Aumentar o intervalo de predicao.
3. Retornar `confidence_flag = low` e explicar a falta de evidencia.
4. Retreinar ou recalibrar a previsao conforme o canal acumular novos posts.

## Data Discovery: recorte confirmado

O TAP exige que cada aluno escolha pelo menos dois entre os setores moda, beleza e automotivo. O recorte deste projeto sera **moda e automotivo**. Beleza permanece fora do primeiro recorte, sem impedir uma expansao futura.

O levantamento inicial encontrou, aplicando idade minima de 30 dias e duracao de ate 180 segundos:

| Setor | Canais com ao menos 30 Shorts | Shorts elegiveis | Observacao |
| --- | ---: | ---: | --- |
| Moda | 6 | 958 | Marcas e criadores; concentracao maior em C&A e EstiloRenner |
| Automotivo | 14 | 1.473 | Marcas e criadores; distribuicao mais ampla entre canais |
| Total do recorte | 20 | 2.431 | Base inicial para a proxima etapa de consolidacao |

Os canais sao classificados por `channel_type` (`brand` ou `creator`) para que diferencas de audiencia nao sejam confundidas com efeito do criativo.

### Canais do piloto

O piloto tera seis canais por setor: tres marcas e tres criadores. Todos atendem ao minimo de 30 Shorts elegiveis.

| Setor | Marcas | Criadores | Shorts no piloto |
| --- | --- | --- | ---: |
| Moda | C&A Brasil, EstiloRenner, Hering Oficial | Steal The Look, Mandy Candy, NiinaSecrets | 958 |
| Automotivo | Hyundai Motor Brasil, Honda Automoveis, Volkswagen do Brasil | Acelerados, Carros com Camanzi, De Carona com Leandro | 944 |
| Total | 6 canais | 6 canais | 1.902 |

Os demais canais aprovados permanecem como reserva para expansao e para testes futuros em canais nao vistos pelo modelo.

## Proxima fase: fechamento da base

Antes do ELT completo, sera feito um levantamento de metadados para escolher o recorte inicial com evidencia. Essa fase deve responder:

- qual setor possui maior volume, frequencia e qualidade de dados;
- quais canais possuem historico suficiente;
- qual periodo historico sera usado;
- qual distribuicao de tamanhos de canal esta disponivel;
- se a distribuicao de curtidas tem variacao suficiente;
- se algum canal domina indevidamente o dataset;
- se o recorte tem dados suficientes para o MVP.

Parametros iniciais a validar no levantamento:

| Item | Hipotese inicial |
| --- | --- |
| Setores do MVP | Moda e automotivo, conforme exigencia de pelo menos dois setores no TAP |
| Canais no MVP | 15 a 25 canais aprovados |
| Videos no MVP | 1.500 a 3.000 Shorts |
| Historico minimo por canal | 30 Shorts elegiveis; preferencia por 50 ou mais |
| Idade minima dos posts | 30 dias, para estabilizar metricas |
| Dominancia de canal | Nenhum canal deve ultrapassar 10% a 15% da base final |
| Meta ampla do TAP | 5.000 a 15.000 videos e 60 ou mais canais |

## Proxima fase: decisoes de modelagem

A parte primaria esta fechada. As decisoes abaixo pertencem ao experimento de modelagem e nao bloqueiam a consolidacao do piloto:

1. Comparar um modelo global com a feature `sector` contra modelos especializados por setor.
2. Definir o alvo de treino principal entre `log(1 + curtidas)` e desempenho relativo ao historico do canal.
3. Definir os cortes temporais de treino, validacao e teste.
4. Definir a primeira versao de features: metadados e texto antes da camada multimodal.

## Referencia

As premissas de escopo, fontes de dados, tecnologias e meta ampla de volume foram extraidas do TAP "TAPI - Inteli Academy x Galaxies".
