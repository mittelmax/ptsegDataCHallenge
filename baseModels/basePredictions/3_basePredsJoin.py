from ptsegDataChallenge.config import data_dir
import pandas as pd

# Reading base models predictions
preds_lgbm = pd.read_csv(f"{data_dir}/basePreds/basePredsLightGBM.csv")
preds_xgb = pd.read_csv(f"{data_dir}/basePreds/basePredsXGBoost.csv")
preds_cat = pd.read_csv(f"{data_dir}/basePreds/basePredsCatboost.csv")
preds_bern = pd.read_csv(f"{data_dir}/basePreds/basePredsBernNB.csv")
preds_logreg = pd.read_csv(f"{data_dir}/basePreds/basePredsLogreg.csv")
preds_knn = pd.read_csv(f"{data_dir}/basePreds/basePredsKNN.csv")
preds_svc = pd.read_csv(f"{data_dir}/basePreds/basePredsSVC.csv")
preds_rf = pd.read_csv(f"{data_dir}/basePreds/basePredsRandomForest.csv")

# Joining level 0 predictions
preds = pd.concat([preds_lgbm, preds_cat.drop("id", axis=1),
                  preds_xgb.drop("id", axis=1),
                  preds_bern.drop("id", axis=1),
                  preds_logreg.drop("id", axis=1),
                  preds_rf.drop("id", axis=1),
                  preds_svc.drop("id", axis=1),
                  preds_knn.drop("id", axis=1)], axis=1)

preds["id"] = preds["id"].astype("int64")  # Converting id column to int

# Saving dataframe
preds.to_csv(f"{data_dir}/basePreds/basePredsJoined.csv", index=False)
