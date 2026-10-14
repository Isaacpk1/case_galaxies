# Metricas e diagnosticos do baseline

## Regra de avaliacao

As metricas sao calculadas no conjunto de teste temporal: os 20% mais recentes de cada canal. A metrica principal e agregada por canal, dando o mesmo peso a cada um dos 12 canais, independentemente de quantos Shorts publicou.

## Metricas de desempenho

| Metrica | Calculo | Uso | Direcao desejada |
| --- | --- | --- | --- |
| MAE macro | Media do MAE de `target_log_likes` entre canais | Metrica primaria para escolher B0, M1, M2 ou M3 | Menor |
| RMSE macro | Media do RMSE de `target_log_likes` entre canais | Penalizar erros grandes | Menor |
| Vies macro | Media de `previsao - resultado` entre canais | Identificar superestimativa ou subestimativa sistematica | Mais proximo de zero |
| Acuracia de ranking por pares | Proporcao de pares de videos do mesmo canal cuja ordem prevista coincide com a ordem observada | Verificar aderencia ao caso de uso de priorizacao de criativos | Maior |
| MAE por canal | MAE calculado separadamente em cada canal | Verificar se um modelo prejudica canais especificos | Menor |
| MAE por tipo de canal | Resumo dos erros de marcas e criadores | Monitorar diferencas entre grupos | Menor e equilibrado |

`target_log_likes = log(1 + like_count)` e o alvo da primeira rodada. Ele reduz a assimetria das curtidas; por isso MAE, RMSE e vies estao na escala logaritmica, nao em numero bruto de curtidas.

## Metricas para comparar hipoteses

| Comparacao | Pergunta | Evidencia |
| --- | --- | --- |
| M1 - B0 | O modelo global acrescenta valor sobre a mediana historica? | Diferenca de MAE macro, bootstrap por canal e teste de permutacao pareado |
| M2 - M1 | Separar marcas e criadores melhora a previsao? | Diferenca de MAE macro, bootstrap por canal e teste de permutacao pareado |
| M1 - M3 | `channel_type` acrescenta sinal alem das outras features? | Diferenca de MAE macro, bootstrap por canal e teste de permutacao pareado |

Valores negativos na diferenca favorecem o primeiro modelo. Como a unidade independente mais proxima e o canal, bootstrap e permutacao reamostram os 12 canais, e nao os 375 videos individualmente.

## Diagnosticos visuais

Os arquivos sao produzidos em `artifacts/baseline/figures/` ao executar `scripts/experimentos/run_baseline_experiments.py`.

| Grafico | Pergunta que responde | Sinal de atencao |
| --- | --- | --- |
| MAE por modelo | Qual opcao tem menor erro medio por canal? | Ganho pequeno com intervalo estatistico amplo |
| Importancia por permutacao de M1 | De quais variaveis o modelo depende para reduzir erro? | Importancia alta nao implica causalidade; features correlacionadas dividem sinal |
| Previsao versus resultado | As previsoes acompanham o resultado e ha faixa sistematicamente errada? | Pontos muito distantes da diagonal ou compressao dos valores altos |
| Erro por canal: M1 versus B0 | Onde M1 melhora ou piora o baseline? | Ganho agregado que esconde piora em varios canais |
| Residuos por video no tempo | O erro muda ao longo do periodo de teste? | Sequencias de residuos positivos ou negativos, indicando drift temporal |
| Distribuicao de erros por tipo | Marcas e criadores recebem qualidade semelhante? | Um grupo com dispersao ou nivel de erro muito maior |

## Limites de interpretacao

- A importancia por permutacao mede dependencia preditiva do modelo M1, nao efeito causal de uma variavel.
- `channel_id` pode aparecer importante porque representa contexto historico do canal; isso nao significa que o modelo funcione em canais completamente novos.
- Graficos e metricas deste baseline servem para decidir o proximo experimento. Eles nao constituem uma promessa de performance em producao.
