from ptsegDataChallenge.config import data_dir
from sklearn.metrics import f1_score
import lightgbm as lgb
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
X_test = pd.read_csv(f"{data_dir}/preproc/testPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsLightGBM.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterLightGBM.pickle"))

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27

# Creating lgbm dataset
train = lgb.Dataset(X, label=y)


# Custom F1 metric for lightGBM
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.5, 0, 1)
    return "f1", f1_score(y_true, y_hat), True


# Setting up parameters
params = {"objective": "binary",
          "boosting": "gbdt",
          "seed": SEED,
          "metric": "f1",
          "is_unbalance": True, **best_params}

# Training model on all data
final_model = lgb.train(params, train, best_iter)

# Lightgbm prediction
preds = final_model.predict(X_test).round(0).astype(int)
preds = pd.Series(preds, name="predicted")
preds_prob = final_model.predict(X_test)
preds_prob = pd.Series(preds_prob, name="predicted_lgbm")

# Creating prediction dataframe
ids_test = X_test["id"]
preds_df = pd.concat([ids_test, preds], axis=1)
preds_prob_df = pd.concat([ids_test, preds_prob], axis=1)

# Saving submission
preds_prob_df.to_csv(f"{data_dir}/basePreds/basePredsLightGBM.csv", index=False)
