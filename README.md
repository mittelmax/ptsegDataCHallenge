# Porto Seguro Data Challenge

<p align="center">
  <img src="https://pngimage.net/wp-content/uploads/2018/06/logo-porto-seguro-png-3.png?raw=true" alt="Logo Porto Seguro"/>
</p>

## `Introdução`

Este repositório contém minha solução para o Porto Seguro Data Challenge, que terminou a competetição em terceiro lugar na divisão universitária. A página e leaderboard oficiais estão disponíveis no Kaggle através deste [link](https://www.kaggle.com/c/porto-seguro-data-challenge). Gostaria de ressaltar o grande aprendizado que esta competição me proporcionou, uma vez que busquei aprender e utilizar técnicas fora de minha zona de conforto. Por fim também gostaria de parabenizar a todos os demais participantes por suas soluções, e também à Porto Seguro pela oportunidade proporcionada.

## `Metodologia e Objetivo`
Meu foco principal durante o desafio foi o de implementar uma estratégia de Stacking. Esta foi a primeira vez que me aventurei com Ensemble Learning, e portanto busquei utilizar tal oportunidade como um incentivo para me familiarizar com tal técnica.

## `Estrutura do Repositório`
Aqui está um resumo da estrutura deste repositório. Minha solução original foi construída com base numa série de notebooks do Kaggle. Dessa forma, este repositório é uma implementação alternativa de minha solução, em que me empenhei para tornar mais organizada e reprodutível.

* `ptsegDataChallenge/`
  * `ptsegDataChallenge/`
    * __init__.py
    * config.py
  * `data/`
    * `raw/`
    * `preproc/`
    * `tuningBase/`
    * `tuningStacked/`
    * `basePreds/`
    * `finalPredictions/`
  * `dataPreprocessing/`
    * preprocV1.py
  * `baseModels/`
    * `tuning/`
      * tuningCatboostBase.py
      * tuningLightGBMBase.py
      * tuningXGBoostBase.py
    * `oofPredictions/`
      * oofPredsBernNB.py
      * oofPredsCatboost.py
      * oofPredsJoin.py
      * oofPredsKNN.py
      * oofPredsLightGBM.py
      * oofPredsLogreg.py
      * oofPredsRandomForest.py
      * oofPredsSVC.py
      * oofPredsXGBoost.py
    * `basePredictions/`
      * basePredsBernNB.py
      * basePredsCatboost.py
      * basePredsJoin.py
      * basePredsKNN.py
      * basePredsLightGBM.py
      * basePredsLogreg.py
      * basePredsRandomForest.py
      * basePredsSVC.py
      * basePredsXGBoost.py
  * `stackedModel/`
      * predictionStackedLightGBM.py
      * tuningStackedLightGBM.py
