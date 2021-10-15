from ptsegDataChallenge.config import data_dir
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import pickle

# Reading data and tuned parameters
df = pd.read_csv(f"{data_dir}/oofPreds/oofPredsJoined.csv")


# Custom F1 metric for lightGBM
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.50, 0, 1)
    return "f1", f1_score(y_true, y_hat), True


# Variables to store tuning information
num_rounds = {}  # Dictionary to store number of iterations @ each round
score_rounds = {}  # Dictionary to store score of each round
count = 0  # Counter to track of iteration number


# Objective function for Optuna
# This function executes lgbm.cv() for several random seeds to prevent overfitting
def objective(trial, seeds=[28, 194, 193, 405]):

    global count  # To access outer scope counter
    scores = []  # List to store score for each seed
    iterations = []  # List to store best numer of iterations for each seed

    # Creating lightGBM datasets every iteration to prevent problems with "min_child_samples" tuning
    X = df.drop(["y"], axis=1)
    y = df["y"]
    train = lgb.Dataset(X, label=y)

    # Defining hyperparameter space for tuning
    params = {
        "objective": "binary",
        "metric": "f1",
        "boosting": "gbdt",
        "is_unbalance": True,
        "verbose": -1,
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 50, 200),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 70),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.08)}

    for seed in seeds:
        # LightGBM training
        num_round = 1000
        modelcv = lgb.cv(params,
                         train,
                         num_round,
                         early_stopping_rounds=40,
                         return_cvbooster=True,
                         feval=lgb_f1_score,
                         nfold=10,
                         verbose_eval=False,
                         seed=seed)

        # Getting best iteration and best score from cross-validation
        best_iteration = modelcv["cvbooster"].best_iteration
        best_score = max(modelcv["f1-mean"])
        scores.append(best_score)
        iterations.append(best_iteration)

    # Best Score
    mean_scores = sum(scores) / len(scores)
    mean_iterations = sum(iterations) / len(iterations)
    print(f"Best Mean F1 Score is: {mean_scores} at mean iteration {mean_iterations}")
    print(f"Best scores: {scores}")
    print(f"Best iterations: {iterations}")
    num_rounds[count] = mean_iterations
    score_rounds[count] = mean_scores
    count += 1
    return mean_scores


# Optuna results
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("Number of finished trials:", len(study.trials))
print(f"Best Mean F1 Score: {score_rounds[study.best_trial.number]} at iterations {num_rounds[study.best_trial.number]}")
print(f"Best Params: {study.best_trial.params}")

# Saving best params
best_params = study.best_trial.params
filename = f"{data_dir}/tuningStacked/bestParamsStacked.pickle"
outfile = open(filename, "wb")
pickle.dump(best_params, outfile)
outfile.close()

best_iter = num_rounds[study.best_trial.number]
filename = f"{data_dir}/tuningStacked/bestIterStacked.pickle"
outfile = open(filename, "wb")
pickle.dump(best_iter, outfile)
outfile.close()
