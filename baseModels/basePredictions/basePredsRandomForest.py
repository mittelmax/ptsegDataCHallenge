from ptsegDataChallenge.config import data_dir
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")
df = df.replace({np.nan: -999})  # BernNB does not handle nan values
X_test = pd.read_csv(f"{data_dir}/preproc/testPreprocV1.csv")
X_test = X_test.replace({np.nan: -999})  # BernNB does not handle nan values

# Splitting train and test datasets
X_train = df.drop("y", axis=1)
y_train = df["y"]

# Final prediction
model = RandomForestClassifier(random_state=27, max_depth=20)
model = model.fit(X_train, y_train)
preds = pd.DataFrame(model.predict_proba(X_test))[1]
preds = pd.Series(preds, name="predicted_rf")

ids_test = X_test["id"].values
ids_test = pd.Series(ids_test, name="id")
preds = pd.concat([ids_test, preds], axis=1)

# Saving submission
preds.to_csv(f"{data_dir}/basePreds/basePredsRandomForest.csv", index=False)
