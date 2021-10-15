from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsLightGBM.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterLightGBM.pickle"))

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27


# Defining custom F1-Score function for models
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

# List to store each of the predicted folds
oof_preds = []

# Initializing Kfold with 10 splits
cv = KFold(n_splits=20, shuffle=True, random_state=SEED)

# Loop to generate oof predictions
for train_idx, test_idx in cv.split(X):

    # Generating lgbm datasets
    train = lgb.Dataset(X.values[train_idx], label=y.values[train_idx])
    test = X.values[test_idx]

    # Training model
    model_lgbm = lgb.train(params, train, num_boost_round=best_iter, feval=lgb_f1_score)

    # Oof Prediction
    preds_lgbm = model_lgbm.predict(test)
    preds_lgbm = pd.Series(preds_lgbm, name="predicted_lgbm")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_lgbm = pd.concat([ids_test, preds_lgbm], axis=1)
    oof_preds.append(preds_lgbm)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsLightGBM.csv", index=False)
