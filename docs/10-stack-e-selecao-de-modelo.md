# ADR-001: Stack do MVP e selecao de hiperparametros

**Status:** Aceita

**Data:** 2026-09-29

**Decisores:** Time Inteli Academy / Galaxies

## Contexto

O projeto precisa provar, em poucas semanas, que atributos de metadados, texto, thumbnail e transcricao melhoram a previsao de performance de YouTube Shorts. O volume inicial e moderado, a avaliacao precisa respeitar o tempo e o resultado deve ser rastreavel por `run_id`, versao de dados e conjunto de features.

O risco principal nao e falta de infraestrutura; e vazamento temporal e uma escolha de modelo baseada no mesmo periodo usado para teste. Por isso a stack deve ser leve e a busca de hiperparametros deve respeitar o rolling-origin.

## Decisao

Usar Python local como ambiente principal, arquivos JSON e Parquet como camadas de dados, DuckDB para consulta analitica local e CatBoost como modelo candidato principal. PyCaret fica restrito a exploracao inicial de modelos; a decisao final usa CatBoost com busca aleatoria dentro de cortes temporais.

```text
Python CLI -> JSON / Parquet / DuckDB -> CatBoost -> SHAP -> FastAPI + FastMCP -> LangGraph
```

## Stack escolhida

| Camada | Tecnologia | Papel no MVP | Status |
| --- | --- | --- | --- |
| Linguagem | Python 3.11+ | Pipeline, modelagem e tool | Em uso |
| Coleta | `requests`, YouTube Data API, `youtube-transcript-api` | Metadados, metricas e transcricoes | Em uso |
| Dados | JSON Bronze, Parquet Silver/Gold, DuckDB | Reprodutibilidade e consultas locais | Parquet/JSON em uso; DuckDB proximo passo |
| Transformacao | pandas, NumPy, PyArrow | Limpeza e materializacao | pandas/NumPy em uso; PyArrow proximo passo |
| Texto | sentence-transformers multilíngue | Embeddings de titulo, descricao e fala | Proximo experimento |
| Imagem | Pillow, OpenCV, CLIP/SigLIP | Atributos e embedding da thumbnail | Proximo experimento |
| Modelo | CatBoost | Regressao tabular com categorias | Decisao para experimento multimodal |
| Intervalos | CatBoost quantile ou MAPIE | Limites de previsao | Proximo experimento |
| Explicabilidade | SHAP | `top_drivers` e auditoria | Proximo experimento |
| Experimentos | MLflow local | Parametros, metricas e artefatos | Proximo passo |
| Exposicao | FastAPI, Pydantic, FastMCP | Funcao/API e tool consumivel | Fase de integracao |
| Orquestracao | LangGraph | Coordenar extracao, previsao e explicacao | Fase de integracao |
| Qualidade | pytest e Ruff | Contratos e regressao | Proximo passo |
| GPU eventual | Colab/Kaggle | Embeddings visuais pesados | Sob demanda |

Nao entram no MVP: microservicos, Kafka, banco vetorial, banco em nuvem ou Kubernetes. Eles so devem ser reavaliados quando volume, latencia ou integracao real exigirem.

## Opcoes consideradas

### Opcao A: PyCaret como AutoML principal

| Dimensao | Avaliacao |
| --- | --- |
| Complexidade | Baixa |
| Velocidade inicial | Alta |
| Controle temporal | Medio, exige configuracao cuidadosa |
| Rastreabilidade | Media |

**Vantagens:** compara modelos e faz tuning rapidamente.

**Limites:** defaults de validacao podem ser inadequados ao rolling-origin; abstrai detalhes importantes para diagnosticar vazamento e dificulta reproduzir a configuracao final com clareza.

### Opcao B: CatBoost + busca aleatoria temporal

| Dimensao | Avaliacao |
| --- | --- |
| Complexidade | Media |
| Velocidade inicial | Media |
| Controle temporal | Alto |
| Rastreabilidade | Alta |

**Vantagens:** lida bem com categorias como canal, setor e tipo de canal; permite registrar cada tentativa; preserva o splitter temporal; e oferece caminho natural para quantis.

**Limites:** requer scripts explicitos e demora mais que uma comparacao AutoML.

### Opcao C: Grid Search exaustivo

| Dimensao | Avaliacao |
| --- | --- |
| Complexidade | Baixa |
| Custo computacional | Alto |
| Cobertura de espaco | Ineficiente |
| Adequacao ao MVP | Baixa |

**Limite principal:** o numero de combinacoes cresce rapidamente e consome folds temporais em pontos pouco promissores.

## Politica de selecao de modelo

1. **Exploracao:** PyCaret pode comparar um conjunto pequeno de candidatos sob o mesmo split temporal: regressao linear, ExtraTrees, Random Forest, LightGBM e CatBoost.
2. **Candidato principal:** CatBoost avanca para tuning se for competitivo em MAE log e Spearman.
3. **Tuning:** executar 30 a 50 tentativas de busca aleatoria dentro dos folds rolling-origin. O periodo futuro reservado nao participa da busca.
4. **Escolha:** ordenar por MAE em log; desempatar por Spearman intra-canal, acuracia par-a-par e estabilidade entre cortes.
5. **Teste final:** treinar uma vez com a configuracao congelada e avaliar no periodo futuro reservado.

O grid inicial de CatBoost e:

```python
SEARCH_SPACE = {
    "depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "l2_leaf_reg": [1, 3, 5, 10, 20],
    "iterations": [300, 500, 800, 1200],
    "random_strength": [0.1, 0.5, 1, 2],
    "bagging_temperature": [0, 0.5, 1, 2],
}
```

O primeiro mecanismo e busca aleatoria. Optuna so deve ser introduzido se 30 a 50 tentativas nao forem suficientes ou se o tempo de treinamento tornar a busca aleatoria cara.

## Consequencias

- Fica simples reproduzir a coleta, o dataset e cada experimento localmente.
- A equipe consegue justificar o modelo final sem depender de uma caixa-preta AutoML.
- A comparacao de modelos continua rapida na fase exploratoria.
- O projeto precisara implementar um splitter rolling-origin reutilizavel antes do tuning final.
- Dependencias planejadas devem ser adicionadas ao ambiente apenas quando o experimento correspondente for iniciado; o `requirements.txt` atual representa o que ja e executavel.

## Acoes

1. [ ] Criar um splitter rolling-origin reutilizavel para os experimentos.
2. [ ] Adicionar CatBoost e MLflow quando iniciar E-03/E-04.
3. [ ] Rodar exploracao limitada com PyCaret sob o split temporal fixado.
4. [ ] Rodar busca aleatoria temporal para o CatBoost.
5. [ ] Congelar configuracao, feature sets e periodo de teste antes da avaliacao final.
