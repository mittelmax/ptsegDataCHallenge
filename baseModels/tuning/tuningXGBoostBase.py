from ptsegDataChallenge.config import data_dir
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import optuna

# Reading data
df = pd.read_csv(f"{data_dir}/preproc/trainPreprocV1.csv")

# Creating XGBoost datasets
X = df.drop("y", axis=1)
y = df["y"]
train = xgb.DMatrix(X, label=y)


# Defining custom F1 metric for XGBoost
def xgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.50, 0, 1)
    return "f1", f1_score(y_true, y_hat)


# Variables to store tuning information
iterations = []  # List to store number of iterations @ each round


# Objective Function
def objective(trial):

    # Setting up hyperparameter space for XGBoost
    params = {"seed": 27,
              "disable_default_eval_metric": True,
              "max_depth": trial.suggest_int("max_depth", 1, 4),
              "gamma": trial.suggest_loguniform("gamma", 1e-8, 2.0),
              "alpha": trial.suggest_loguniform("alpha", 1e-8, 2.0),
              "min_child_weight": trial.suggest_uniform("min_child_weight", 0.5, 1.5),
              "eta": trial.suggest_uniform("eta", 0.20, 0.26),
              "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1)}

    model_cv = xgb.cv(params,
                      train,
                      feval=xgb_f1_score,
                      num_boost_round=1000,
                      stratified=True, seed=27,
                      maximize=True,
                      verbose_eval=False,
                      early_stopping_rounds=30,
                      nfold=5)

    best_score = model_cv["test-f1-mean"].max()
    best_iteration = model_cv[model_cv["test-f1-mean"] == best_score].index.values[0]
    iterations.append(best_iteration)
    print(f"Best F1-Score is {best_score} at iteration {best_iteration}")
    return best_score


# Running Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)
print("Number of finished trials:", len(study.trials))
print(f"Best is trial {study.best_trial.number} with score {study.best_trial.value} and parameters {study.best_trial.params}")

# Saving best params
best_params = study.best_trial.params
filename = f"{data_dir}/tuningBase/bestParamsXGBoost.pickle"
outfile = open(filename, "wb")
pickle.dump(best_params, outfile)
outfile.close()

best_iter = iterations[study.best_trial.number]
filename = f"{data_dir}/tuningBase/bestIterXGBoost.pickle"
outfile = open(filename, "wb")
pickle.dump(best_iter, outfile)
outfile.close()
