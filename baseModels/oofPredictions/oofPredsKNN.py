from ptsegDataChallenge.config import data_dir
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
df = df.replace({np.nan: -999})  # KNN does not handle nan values

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
    model_knn = KNeighborsClassifier()
    model_knn = model_knn.fit(X_train, y_train)

    # Oof Prediction
    preds_knn = pd.DataFrame(model_knn.predict_proba(X_test))[1]
    preds_knn = pd.Series(preds_knn, name="predicted_knn")

    # Generating dataset for meta model
    # Creating prediction dataframe
    ids_test = X["id"].values[test_idx]
    ids_test = pd.Series(ids_test, name="id")

    preds_knn = pd.concat([ids_test, preds_knn], axis=1)
    oof_preds.append(preds_knn)

# Concatenating predictions
final_df = pd.concat(oof_preds).sort_values("id")
final_df = final_df.set_index("id").reset_index()

# Saving oof predictions
final_df.to_csv(f"{data_dir}/oofPreds/oofPredsKNN.csv", index=False)
