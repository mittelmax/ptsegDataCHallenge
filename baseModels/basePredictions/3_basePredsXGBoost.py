from ptsegDataChallenge.config import data_dir
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
X_test = pd.read_csv(f"{data_dir}/preproc/testPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsXGBoost.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterXGBoost.pickle"))

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27

# Creating XGBoost datasets
train = xgb.DMatrix(X, label=y)
test = xgb.DMatrix(X_test)


# Custom F1 metric for XGBoost
def xgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.50, 0, 1)
    return "f1", f1_score(y_true, y_hat)


# Setting up parameters
params = {"seed": SEED, **best_params}

# Final prediction
model = xgb.train(params, train, num_boost_round=best_iter, feval=xgb_f1_score, maximize=True)
preds = model.predict(test)
preds = pd.Series(preds, name="predicted")

# Creating prediction dataframe
ids_test = X_test["id"]
preds_df = pd.concat([ids_test, preds], axis=1)

# Saving submission
preds_df.to_csv(f"{data_dir}/basePreds/basePredsXGBoost.csv", index=False)
