from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
df = df.replace({np.nan: -999})  # Logistic Regression does not handle nan values

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
    model_rf = RandomForestClassifier(random_state=SEED, max_depth=20)
    model_rf = model_rf.fit(X_train, y_train)

    # Oof Prediction
    preds_rf = pd.DataFrame(model_rf.predict_proba(X_test))[1]
    preds_rf = pd.Series(preds_rf, name="predicted_rf")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_rf = pd.concat([ids_test, preds_rf], axis=1)
    oof_preds.append(preds_rf)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsRandomForest.csv", index=False)
