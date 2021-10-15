from ptsegDataChallenge.config import data_dir
from catboost import Pool
from catboost import cv
import numpy as np
import pandas as pd
import pickle
import optuna

# # # Disclaimer # # #
# For unknown reasons Catboost versions above 0.26 cannot use class weights with a custom loss metric
# Make sure to downgrade Catboost to 0.26 to execute this script
# # # # # # # # # # # #

# Reading data
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")

# Calculatig weight of classes
ocurrences = df["y"].value_counts()
weight_pos = ocurrences[1]/(ocurrences[0]+ocurrences[1])
weight_neg = 1 - weight_pos

# Defining which features should be treated as categoricals by catboost
# This combination of features was determined by comparing F1 scores across # combinations
cat_features = ["var29", "var30", "var34", "var37"]
df[cat_features] = df[cat_features].replace({np.nan: "NA"})
df[cat_features] = df[cat_features].astype(str)

# Splitting datasets
X = df.drop("y", axis=1)
y = df["y"]

# Creating catboost Pool
train_pool = Pool(data=X, label=y, has_header=True, cat_features=cat_features)

# Variables to store tuning information
iterations = []  # List to store number of iterations @ each round


# Objective Function
def objective(trial):

    params = {"random_seed": 27,
              "custom_loss": "F1",
              "eval_metric": "F1",
              "loss_function": "Logloss",
              "class_weights": (weight_pos, weight_neg),
              "num_boost_round": 1000,
              "early_stopping_rounds": 10,
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
              "depth": trial.suggest_int("depth", 3, 10),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
              "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
              "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 0, 8),
              "verbose_eval": False}

    # Cross validation
    cv_data = cv(params=params, pool=train_pool, fold_count=5, shuffle=True, partition_random_seed=27,)

    # Best Score and Best Iteration
    best_score = cv_data["test-F1:use_weights=false-mean"].max()
    best_iteration = cv_data[cv_data["test-F1:use_weights=false-mean"] == best_score].index.values[0]
    iterations.append(best_iteration)

    # Best Score without tuning
    print(f"Best F1 Score is: {best_score} at iteration {best_iteration}")
    print("Trial done: F1 values on folds: {}".format(best_score))
    return best_score


# Optuna results
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=400)
print("Number of finished trials:", len(study.trials))
print("Best trial:", study.best_trial.params)

# Saving best params
best_params = study.best_trial.params
filename = f"{data_dir}/tuningBase/bestParamsCatboost.pickle"
outfile = open(filename, "wb")
pickle.dump(best_params, outfile)
outfile.close()

best_iter = iterations[study.best_trial.number]
filename = f"{data_dir}/tuningBase/bestIterCatboost.pickle"
outfile = open(filename, "wb")
pickle.dump(best_iter, outfile)
outfile.close()
