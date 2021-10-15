from ptsegDataChallenge.config import data_dir
import pandas as pd

# Reading true y value
y_true = pd.read_csv(f"{data_dir}/raw/train.csv", usecols=["id", "y"])

# Reading oof models predictions
preds_lgbm = pd.read_csv(f"{data_dir}/oofPreds/oofPredsLightGBM.csv")
preds_xgb = pd.read_csv(f"{data_dir}/oofPreds/oofPredsXGBoost.csv")
preds_cat = pd.read_csv(f"{data_dir}/oofPreds/oofPredsCatboost.csv")
preds_bern = pd.read_csv(f"{data_dir}/oofPreds/oofPredsBernNB.csv")
preds_logreg = pd.read_csv(f"{data_dir}/oofPreds/oofPredsLogreg.csv")
preds_knn = pd.read_csv(f"{data_dir}/oofPreds/oofPredsKNN.csv")
preds_svc = pd.read_csv(f"{data_dir}/oofPreds/oofPredsSVC.csv")
preds_rf = pd.read_csv(f"{data_dir}/oofPreds/oofPredsRandomForest.csv")

# Joining oof predictions
preds = pd.concat([preds_lgbm, preds_cat.drop("id", axis=1),
                  preds_xgb.drop("id", axis=1),
                  preds_bern.drop("id", axis=1),
                  preds_logreg.drop("id", axis=1),
                  preds_rf.drop("id", axis=1),
                  preds_svc.drop("id", axis=1),
                  preds_knn.drop("id", axis=1)], axis=1)

preds["id"] = preds["id"].astype("int64")  # Converting id column to int

# Joining predictions with true values
final_df = preds.merge(y_true, how="inner")
final_df = final_df.set_index("id").reindex(y_true["id"])
final_df = final_df.reset_index()

# Saving dataframe
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsJoined.csv", index=False)
