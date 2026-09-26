# Hipoteses de modelagem do piloto

## Pergunta de decisao

Os canais de marcas e criadores (influenciadores) devem alimentar um unico modelo global ou dois modelos globais especializados por `channel_type`?

Nesta fase, **modelo global** significa um modelo treinado com varios canais; nao significa um modelo individual por canal. Um modelo por canal nao sera testado no primeiro experimento, porque cada canal teria apenas 26 a 218 observacoes de treino depois do corte temporal.

## Hipoteses

### H0 - A mediana historica e dificil de superar

- **Hipotese nula**: nenhum modelo tabular supera de forma consistente `baseline_log_likes`, a mediana de curtidas dos posts anteriores do mesmo canal.
- **Por que importa**: o projeto so deve adotar um modelo se ele agregar valor sobre uma regra simples e explicavel.
- **Teste**: comparar MAE e RMSE em `target_log_likes` no conjunto de teste contra `baseline_log_likes`.

### H1 - Um unico modelo global generaliza melhor

- **Hipotese**: um modelo unico, treinado com marcas e criadores e recebendo `channel_type`, `sector` e historico do canal como contexto, tera melhor erro medio por canal.
- **Racional**: ele usa todas as 1.467 linhas de treino e pode aprender padroes de formato, horario e frequencia compartilhados entre tipos de canal.
- **Risco**: a maior quantidade de marcas na base pode fazer o modelo favorecer esse grupo.

### H2 - Modelos especializados por tipo de canal generalizam melhor dentro do grupo

- **Hipotese concorrente**: dois modelos globais, um de marcas e outro de criadores, terao menor erro medio dentro de seus respectivos grupos.
- **Racional**: audiencia, frequencia de publicacao e distribuicao de curtidas podem ser estruturalmente diferentes entre marcas e criadores.
- **Risco**: os dois modelos terao menos dados de treino; o grupo de criadores e especialmente menor.

### H3 - `channel_type` tem sinal preditivo mesmo apos o historico do canal

- **Hipotese**: incluir `channel_type` melhora o modelo global em relacao ao mesmo modelo sem essa feature.
- **Racional**: se a diferenca entre marca e criador persiste apos controlar pela mediana historica do canal, ela deve aparecer no erro de previsao.
- **Interpretação**: se nao houver ganho, a separacao em dois modelos perde justificativa inicial.

### H4 - A especializacao por setor deve ser avaliada depois do tipo de canal

- **Hipotese**: modelos separados de moda e automotivo podem melhorar a previsao, mas essa decisao sera testada somente apos decidir entre H1 e H2.
- **Racional**: cruzar setor e tipo de canal agora criaria quatro modelos pequenos e reduziria demais a evidencia disponivel.

## Experimentos controlados

Todos os experimentos usam as mesmas 1.467 linhas de treino e as mesmas 375 linhas de teste, definidas como os 20% mais recentes de cada canal. O preprocessamento deve ser ajustado somente no treino.

| Codigo | Modelo | Dados de treino | Predicao no teste |
| --- | --- | --- | --- |
| B0 | Mediana historica por canal | Historico anterior do proprio canal | Todas as linhas com `baseline_log_likes` |
| M1 | Modelo global com `channel_type` | Marcas + criadores | Todas as linhas |
| M2 | Dois modelos globais por `channel_type` | Um modelo para marcas; outro para criadores | Cada linha usa apenas o modelo do seu tipo |
| M3 | Modelo global sem `channel_type` | Marcas + criadores | Todas as linhas |

## Criterio de decisao

1. Eliminar qualquer modelo que nao supere B0 no MAE medio por canal do teste.
2. Comparar M1 e M2 pelo MAE medio por canal, nao apenas pelo erro agregado por video. Isso impede que os canais com mais Shorts decidam sozinhos a escolha.
3. Conferir MAE por `channel_type`, RMSE e viés medio para garantir que um ganho agregado nao esconda piora relevante em um dos grupos.
4. Se a diferenca entre M1 e M2 for pequena ou instavel em reamostragens por canal, preferir M1: ele e mais simples, usa mais dados e serve a todos os canais.
5. Se M2 vencer de forma consistente dentro de ambos os tipos e superar B0, adotar os dois modelos especializados.

## O que nao fazer no primeiro baseline

Nao devemos treinar modelos por canal nem treinar um modelo de marcas e outro de criadores para depois tirar a media de suas previsoes. Cada um desses modelos especializados deve prever apenas linhas do seu proprio grupo. Uma combinacao de modelos so faria sentido mais tarde como ensemble, com predicoes out-of-fold para evitar vazamento.
