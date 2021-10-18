from ptsegDataChallenge.config import data_dir
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pandas as pd

# Reading data
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")

# Creating datasets
X_train = df.drop("y", axis=1)
features = X_train.columns.values.tolist()  # Saving colnames for later
X_train = X_train.values
y_train = df["y"].values.ravel()

# Training model
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced")

# Setting up boruta
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=27, perc=30)
feat_selector.fit(X_train, y_train)  # Running boruta

# Investigating selected features
features = pd.Series(features, name="feature")
selected_vars = pd.Series(feat_selector.support_.tolist(), name="Selected")
selected_vars = pd.concat([features, selected_vars], axis=1).sort_values("Selected", ascending=False)

# Saving Boruta output:
selected_vars.to_csv(f"{data_dir}/borutaOutput", index=False)
