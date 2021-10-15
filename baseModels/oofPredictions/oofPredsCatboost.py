from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
best_params = pd.read_pickle(f"{data_dir}/tuningBase/bestParamsCatboost.pickle")
best_iter = round(pd.read_pickle(f"{data_dir}/tuningBase/bestIterCatboost.pickle"))

# Setting up data types
cat_features = ["var29", "var30", "var34", "var37"]
df[cat_features] = df[cat_features].replace({np.nan: "NA"})
df[cat_features] = df[cat_features].astype(str)

# Calculatig weight of classes
ocurrences = df["y"].value_counts()
weight_pos = ocurrences[1]/(ocurrences[0]+ocurrences[1])
weight_neg = 1 - weight_pos

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]
cat_idx = [X.columns.get_loc(col) for col in cat_features]

# Defining random seed
SEED = 27

# Setting up Catboost parameters
params = {"random_seed": SEED,
          "custom_loss": "F1",
          "eval_metric": "F1",
          "loss_function": "Logloss",
          "num_boost_round": best_iter,
          "verbose": False,
          "class_weights": (weight_pos, weight_neg), **best_params}

# List to store each of the predicted folds
oof_preds = []

# Initializing Kfold with 10 splits
cv = KFold(n_splits=20, shuffle=True, random_state=SEED)

# Loop to generate oof predictions
for train_idx, test_idx in cv.split(X):

    # Generating datasets
    X_train = X.values[train_idx]
    X_test = X.values[test_idx]
    y_train = y.values[train_idx]

    # Training model
    model_cat = CatBoostClassifier(**params)
    model_cat = model_cat.fit(X_train, y_train, cat_features=cat_idx)

    # Oof Prediction
    preds_cat = pd.DataFrame(model_cat.predict_proba(X_test))[1]
    preds_cat = pd.Series(preds_cat, name="predicted_cat")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_cat = pd.concat([ids_test, preds_cat], axis=1)
    oof_preds.append(preds_cat)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsCatboost.csv", index=False)
