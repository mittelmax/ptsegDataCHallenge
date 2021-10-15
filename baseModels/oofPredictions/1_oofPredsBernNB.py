from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
df = df.replace({np.nan: -999})  # BernNB does not handle nan values

# Splitting dataset
X = df.drop("y", axis=1)
y = df["y"]

# Defining random seed
SEED = 27

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
    model_bern = BernoulliNB(alpha=1.4)
    model_bern = model_bern.fit(X_train, y_train)

    # Oof Prediction
    preds_bern = pd.DataFrame(model_bern.predict_proba(X_test))[1]
    preds_bern = pd.Series(preds_bern, name="predicted_bern")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_bern = pd.concat([ids_test, preds_bern], axis=1)
    oof_preds.append(preds_bern)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsBernNB.csv", index=False)
