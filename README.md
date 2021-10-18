# Porto Seguro Data Challenge

<p align="center">
  <img src="https://pngimage.net/wp-content/uploads/2018/06/logo-porto-seguro-png-3.png?raw=true" alt="Logo Porto Seguro"/>
</p>

## `Introdução`

Este repositório contém minha solução para o Porto Seguro Data Challenge, que terminou a competetição em terceiro lugar na divisão universitária. A página e leaderboard oficiais estão disponíveis no Kaggle através [deste link](https://www.kaggle.com/c/porto-seguro-data-challenge). Gostaria de ressaltar o grande aprendizado que esta competição me proporcionou, uma vez que busquei aprender e utilizar técnicas fora de minha zona de conforto. Por fim também gostaria de parabenizar a todos os demais participantes por suas soluções, e também à Porto Seguro pela oportunidade proporcionada.

## `Metodologia e Objetivo`
Meu foco principal durante o desafio foi o de implementar uma estratégia de Stacking. Esta foi a primeira vez que me aventurei com Ensemble Learning, e portanto busquei utilizar tal oportunidade como um incentivo para me familiarizar com tal técnica. 
\
\
Foram utilizados 8 modelos base:
* `LightGBM`
* `XGBoost`
* `Catboost`
* `Random Forest`
* `Support Vector Classifier`
* `Bernoulli Naive Bayes`
* `Logistic Regression`
* `K Nearest Neighbours Classifier`

E como meta-modelo utilizou-se novamente o `LightGBM`


## `Requisitos`
Para a execução de minhha estratégia utilizei os seguintes pacotes disponíveis no arquivo `requirements.txt`.
* catboost==0.26
* lightgbm==3.3.0
* numpy==1.21.2
* optuna==2.10.0
* pandas==1.3.3
* pathlib==1.0.1
* scikit-learn==1.0
* xgboost==1.4.2

#### `Disclaimer`
As versões mais recentes do Catboost apresentam um problema que impede a utilização do parâmetro `class_weights` com a métrica de Loss F1. Dessa forma, para a execução da solução, é necessário que se realize um downgrade do pacote para a versão 0.26.

## `Estrutura do Repositório`
Aqui está um resumo da estrutura deste repositório. Minha solução original foi construída com base numa série de notebooks do Kaggle. Dessa forma, este repositório é uma implementação alternativa de minha solução, em que me empenhei para tornar mais organizada e reprodutível.
\
\
Os scripts estão enumerados de acordo com a ordem de preferência na execução. Arquivos com o mesmo número inicial podem ser executados paralelamente. Já os scrits referentes à otimização de hiperparâmetros não foram enumerados uma vez que sua execução é opcional, podendo-se utilizar os parâmetros que já estão salvos no repositório.

* `ptsegDataChallenge/`
  * `ptsegDataChallenge/`
    * __init__.py
    * config.py
  * `dataPreprocessing/`
    * 0_preprocV1.py
  * `baseModels/`
    * `tuning/`
      * tuningCatboostBase.py
      * tuningLightGBMBase.py
      * tuningXGBoostBase.py
    * `oofPredictions/`
      * 1_oofPredsBernNB.py
      * 1_oofPredsCatboost.py
      * 1_oofPredsKNN.py
      * 1_oofPredsLightGBM.py
      * 1_oofPredsLogreg.py
      * 1_oofPredsRandomForest.py
      * 1_oofPredsSVC.py
      * 1_oofPredsXGBoost.py
      * 2_oofPredsJoin.py
    * `basePredictions/`
      * 3_basePredsBernNB.py
      * 3_basePredsCatboost.py
      * 3_basePredsKNN.py
      * 3_basePredsLightGBM.py
      * 3_basePredsLogreg.py
      * 3_basePredsRandomForest.py
      * 3_basePredsSVC.py
      * 3_basePredsXGBoost.py
      * 4_basePredsJoin.py
  * `boruta/`
      * featuresBoruta.py
      * tuningBoruta.py
  * `stackedModel/`
      * 5_predictionStackedLightGBM.py
      * tuningStackedLightGBM.py
  * `README.md`
  * `requirements.txt`
  * `data/`
    * `raw/`
    * `preproc/`
    * `tuningBase/`
    * `tuningStacked/`
    * `basePreds/`
    * `finalPredictions/`
   
Agora vamos analisar melhor o funcionamento de cada uma das partes do repositório:
## `ptsegDataChallenge/`
  * `ptsegDataChallenge/`
    * __init__.py
    * config.py

Esta pasta, com o mesmo nome do repositório, atua como um pacote de Pyhton que contém as configurações do projeto. Neste caso específico, o script config.py contém o path do diretório `data/` que deve ser inserido pelo usuário, onde estão alocados os datasets de treino e teste e também arquivos referentes à minha solução.

```python
from pathlib import Path

# Use this variable to insert the path to your data directory
# This variable will be avaiable to all python scripts
data_dir = Path("/path to the data folder")
```

## `dataPreprocessing/`
  * `dataPreprocessing/`
    * 0_preprocV1.py

Esta pasta contém os scripts referentes ao pré-processamento dos dados. Testei muitas técnicas em minha solução sem obter nenhhum ganho de performance. A única estratégia que rendeu alguma melhora foi substituir os valores `-999` por `np.nan`.

## `baseModels`
  * `baseModels/`
    * `tuning/`
      * tuningCatboostBase.py
      * tuningLightGBMBase.py
      * tuningXGBoostBase.py
    * `oofPredictions/`
      * 1_oofPredsBernNB.py
      * 1_oofPredsCatboost.py
      * 1_oofPredsKNN.py
      * 1_oofPredsLightGBM.py
      * 1_oofPredsLogreg.py
      * 1_oofPredsRandomForest.py
      * 1_oofPredsSVC.py
      * 1_oofPredsXGBoost.py
      * 2_oofPredsJoin.py
    * `basePredictions/`
      * 3_basePredsBernNB.py
      * 3_basePredsCatboost.py
      * 3_basePredsKNN.py
      * 3_basePredsLightGBM.py
      * 3_basePredsLogreg.py
      * 3_basePredsRandomForest.py
      * 3_basePredsSVC.py
      * 3_basePredsXGBoost.py
      * 4_basePredsJoin.py

Esta pasta contém todo o código referente aos modelos da camada base de meu ensemble, estando dividida em 3 partes:

* `tuning/`: Contém os scripts referentes à otimização de hiperparâmetros do `XGBoost`, `Catboost` e `LightGBM`
* `oofPredictions/`: Contém os scripts referentes às previsões out-of-fold dos modelos da camada base do ensemble
* `basePredictions/`: Contém os scripts referentes às previsões finais para cada um dos modelos base do ensemble, que serão utilizadas para a previsão final

## `boruta/`
  * `boruta/`
      * featuresBoruta.py
      * tuningBoruta.py

Esta pasta contém scripts relacionados ao procesos de features selection com o pacote Boruta. Como os resultados não foram promissores, acabei não utilizando apenas as variáveis selecionadas pelo boruta no modelo final. De todo modo, decidi deixar tal código disponível aqui.

## `stackedModel`
  * `stackedModel/`
      * 5_predictionStackedLightGBM.py
      * tuningStackedLightGBM.py

Esta pasta contem os scipts de otimização de parâmetros e tambbém da previsão final do meta-modelo de stacking, isto é, o modelo que tem como input as previsões dos modelos base.

## `data/`
  * `data/`
    * `raw/`
    * `preproc/`
    * `tuningBase/`
    * `tuningStacked/`
    * `basePreds/`
    * `finalPredictions/`

É a pasta que contém os inputs e outputs dos modelos. Os dados originais da competição devem ser incluídos na pasta `raw/`. Na pasta `preproc/` armazena-se o output do pré-processamento da base original. Já nas demais pastas, armazena-se informaçÕes referentes à otimização de hiperparâmetros e as previsões dos modelos base e do modelo final.

## `Observação`
Durante o processo de transformação da minha solução final feita em notebooks do Kaggle para este repositório, percebi a existência de um erro no código:
Durante a previsão out-of-fold do `Support Vector Classifier`, acabei utilizando o `Random Forest` como modelo. Dessa forma, na previsão final da competição acabei utilizando duas vezes as previsões do `Random Forest`no modelo final. Este erro está corrigido neste repositório. Observei ganhos de performance ao comparar este modelo revisado com o original.

## `Conclusões`
Sou muito grato à oportunidade de aprendizagem proporcionada por esta competição. Fico muito feliz em ter implementado uma solução de stacking pela primeira vez e também por ter terminado entre os 3 melhores colocados de minha divisão. Deixo aqui novamente meus agradecimentos à Porto Seguro pela oportunidade proporcionada!

