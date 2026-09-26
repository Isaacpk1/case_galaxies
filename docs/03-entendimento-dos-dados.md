# Entendimento dos Dados

## Objetivo

Esta etapa responde se os dados disponiveis representam o problema que queremos prever e se podem ser preparados para um experimento de machine learning sem vazamento de informacao.

O recorte inicial usa YouTube Shorts publicos dos setores de moda e automotivo, publicados por marcas e criadores brasileiros.

## Unidade de analise

Cada linha representa **um video publicado por um canal em uma data e hora especificas**.

O mesmo canal aparece em diversas linhas porque publicou diversos Shorts. Portanto, videos do mesmo canal nao sao observacoes independentes: eles compartilham audiencia, estilo de conteudo e tamanho de canal.

## Origem e recorte

| Item | Definicao |
| --- | --- |
| Fonte | YouTube Data API v3, rota publica |
| Setores | Moda e automotivo |
| Mercado | Brasil |
| Tipos de canal | `brand` e `creator` |
| Janela historica | 18 meses |
| Idade minima do post | 30 dias |
| Duracao usada para inferir Short | 1 a 180 segundos |
| Minimo de historico | 30 Shorts por canal |
| Piloto | 12 canais, 1.902 Shorts |

## Tabelas disponiveis

### Catalogo de canais

Arquivo: `data/candidates/channels.csv`

Define a origem do recorte. Cada linha contem `channel_id`, nome do canal, setor, mercado, tipo do canal e uma nota de curadoria.

### Bronze: resposta bruta da API

Pasta: `data/bronze/youtube/<run_id>/`

Preserva as respostas coletadas da API em JSON. Ela permite refazer as transformacoes sem consultar a API novamente e manter rastreabilidade da coleta.

### Silver: Shorts elegiveis

Arquivo: `data/silver/data_discovery/<run_id>/eligible_shorts.csv`

Contem somente videos que atenderam aos criterios de elegibilidade. Os campos principais sao:

| Grupo | Campos | Significado |
| --- | --- | --- |
| Identificacao | `video_id`, `channel_id`, `channel_name` | Identificadores do video e do canal |
| Contexto | `sector`, `market`, `channel_type` | Segmento e origem do canal |
| Conteudo textual | `title`, `description`, `tags` | Texto publicado junto ao video |
| Tempo e formato | `published_at`, `duration_seconds`, `privacy_status` | Quando e como o video foi publicado |
| Resultados observados | `view_count`, `like_count`, `comment_count` | Metricas publicas coletadas depois da publicacao |
| Midia | `thumbnail_url` | Referencia publica para a thumbnail |
| Rastreabilidade | `candidate_notes`, `collected_at` | Contexto de curadoria e instante da coleta |

### Silver: perfil de canal

Arquivo: `data/silver/data_discovery/<run_id>/channel_profile.csv`

Resume cada canal com volume de Shorts elegiveis, periodo coberto, mediana e media de curtidas, inscritos no momento da coleta e flags de historico minimo/preferencial.

## O que cada metrica significa

| Metrica | Significado | Pode ser usada antes da publicacao? |
| --- | --- | --- |
| `like_count` | Curtidas observadas no video | Nao. E alvo/resultado para treino e avaliacao. |
| `view_count` | Reproducoes observadas no video | Nao. E resultado posterior. |
| `comment_count` | Comentarios observados no video | Nao. E resultado posterior. |
| Inscritos do canal na coleta | Tamanho aproximado do canal no instante da coleta | Sim, com ressalva: nao e o valor historico exato no instante de cada post. |
| Titulo, descricao, tags, duracao e data | Metadados do video | Sim, pois existem quando o video e publicado. |

## Regra central: prevenir vazamento

Para prever a performance de um video, nunca podemos usar `like_count`, `view_count` ou `comment_count` desse mesmo video como feature. Eles sao conhecidos somente apos a publicacao.

Tambem devemos calcular qualquer estatistica de historico do canal usando apenas videos publicados **antes** do video que esta sendo previsto. Por exemplo, a mediana historica de curtidas de um video publicado em junho nao pode incluir videos publicados em julho.

## Qualidade observada

- A base possui marcas e criadores. Essa diferenca deve permanecer explicita no campo `channel_type`.
- Os canais possuem escalas de curtidas muito diferentes; por isso a comparacao absoluta entre canais e limitada.
- Alguns canais candidatos possuem poucos ou nenhum Short elegivel. Eles ficam fora do piloto, mas podem ser acompanhados em coletas futuras.
- O recorte atual tem volume suficiente para um piloto exploratorio, mas nao atinge a meta ampla do TAP de 5.000 a 15.000 videos e 60 ou mais canais.

## Limitacoes conhecidas

1. A API publica nao fornece impressoes, CTR de thumbnail, retencao, tempo assistido ou origem de trafego.
2. As metricas sao uma fotografia no momento da coleta. Um video mais antigo teve mais tempo para acumular curtidas que um video mais recente, mesmo que ambos tenham pelo menos 30 dias.
3. A classificacao como Short e uma aproximacao por duracao de ate 180 segundos; a API nao fornece uma flag publica exclusiva de Short.
4. Ainda nao foram baixados video, audio, frames, transcricoes ou embeddings. Esses dados pertencem a etapa multimodal.
5. O valor de inscritos disponivel e o da coleta, nao necessariamente o valor existente na data de publicacao de cada video.

## Hipoteses para a proxima etapa

1. Usar `log(1 + like_count)` como alvo de regressao reduz a assimetria extrema das curtidas.
2. Testar tambem um alvo relativo ao canal, comparando as curtidas do video com a mediana dos posts anteriores daquele canal.
3. Manter `sector` e `channel_type` como features contextuais e avaliar seu impacto.
4. Validar no tempo: treinar no passado e testar em posts mais recentes.

## Criterio de conclusao desta fase

Data Understanding estara concluida quando o time conseguir responder, com base neste documento:

- o que representa uma linha;
- de onde cada campo veio;
- quais campos sao entrada e quais sao resultado;
- quais limitacoes e vieses existem;
- qual recorte sera preparado para o primeiro baseline.

## Relacao com outros documentos

- [Entendimento do problema](01-entendimento-do-problema.md)
- [Registro de descobertas e ideias](02-registro-de-descobertas.md)
