from ptsegDataChallenge.config import data_dir
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import lightgbm as lgb

# Reading dataframes and tuned parameters
train = pd.read_csv(f"{data_dir}/oofPreds/oofPredsJoined.csv")
X_test = pd.read_csv(f"{data_dir}/basePreds/basePredsJoined.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningStacked/bestParamsStacked.pickle")
best_iter = pd.read_pickle(f"{data_dir}/tuningStacked/bestIterStacked.pickle")

# Splitting target from features
X_train = train.drop(["y"], axis=1)
y_train = train["y"]

# Creating lgbm dataset
train_lgbm = lgb.Dataset(X_train, label=y_train)


# Custom F1 metric for lightGBM
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.5, 0, 1)
    return "f1", f1_score(y_true, y_hat), True


params = {"objective": "binary",
          "boosting": "gbdt",
          "metric": "f1",
          "is_unbalance": True,
          "verbosity": -1, **best_params}

# Training model on all data
final_model = lgb.train(params, train_lgbm, best_iter-3)

# Lightgbm prediction
preds = final_model.predict(X_test).round(0).astype(int)
preds = pd.Series(preds, name="predicted")

# Creating prediction dataframe
ids_test = X_test["id"]
preds_df = pd.concat([ids_test, preds], axis=1)

# Saving predictions for submission
preds_df["id"] = preds_df["id"].astype(int)
preds_df.to_csv(f"{data_dir}/finalPrediction/sumbission.csv", index=False)
