from ptsegDataChallenge.config import data_dir
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
X_test = pd.read_csv(f"{data_dir}/preproc/testPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsCatboost.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterCatboost.pickle"))

# Setting up data types
cat_features = ["var29", "var30", "var34", "var37"]
df[cat_features] = df[cat_features].replace({np.nan: "NA"})
df[cat_features] = df[cat_features].astype(str)
X_test[cat_features] = X_test[cat_features].replace({np.nan: "NA"})
X_test[cat_features] = X_test[cat_features].astype(str)

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27

# Calculatig weight of classes
ocurrences = df["y"].value_counts()
weight_pos = ocurrences[1]/(ocurrences[0]+ocurrences[1])
weight_neg = 1 - weight_pos

# Setting up Catboost parameters
params = {"random_seed": SEED,
          "custom_loss": "F1",
          "eval_metric": "F1",
          "loss_function": "Logloss",
          "num_boost_round": best_iter,
          "verbose": False,
          "class_weights": (weight_pos, weight_neg), **best_params}

# Training on all data
model = CatBoostClassifier(**params)
model = model.fit(X, y, cat_features=cat_features)

# Prediction
preds = pd.DataFrame(model.predict_proba(X_test))[1]
preds = pd.Series(preds, name="predicted_cat")

# Creating prediction dataframe
ids_test = X_test["id"]
preds_df = pd.concat([ids_test, preds], axis=1)

# Saving submission
preds_df.to_csv(f"{data_dir}/basePreds/basePredsCatboost.csv", index=False)
