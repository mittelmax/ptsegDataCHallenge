from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsXGBoost.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterXGBoost.pickle"))

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27


# Defining custom F1-Score function for XGBoost
def xgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.50, 0, 1)
    return "f1", f1_score(y_true, y_hat)


# Setting up parameters
params = {"seed": SEED,
          "disable_default_eval_metric": True, **best_params}

# List to store each of the predicted folds
oof_preds = []

# Initializing Kfold with 10 splits
cv = KFold(n_splits=20, shuffle=True, random_state=SEED)

# Loop to generate oof predictions
for train_idx, test_idx in cv.split(X):

    # Generating datasets
    train = xgb.DMatrix(X.values[train_idx], label=y.values[train_idx])
    test = xgb.DMatrix(X.values[test_idx])

    # Training model
    model_xgb = xgb.train(params, train, num_boost_round=81, feval=xgb_f1_score, maximize=True)

    # Oof Prediction
    preds_xgb = model_xgb.predict(test)
    preds_xgb = pd.Series(preds_xgb, name="predicted_xgb")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_xgb = pd.concat([ids_test, preds_xgb], axis=1)
    oof_preds.append(preds_xgb)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsXGBoost.csv", index=False)
