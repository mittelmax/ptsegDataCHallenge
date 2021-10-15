from ptsegDataChallenge.config import data_dir
import pandas as pd
import numpy as np

# Reading data
df_train = pd.read_csv(f"{data_dir}/raw/train.csv")
df_test = pd.read_csv(f"{data_dir}/raw/test.csv")

# NaN imputation
df_train = df_train.replace({-999: np.nan})
df_test = df_test.replace({-999: np.nan})

# Saving dataframe
df_train.to_csv(f"{data_dir}/preproc/trainPreprocV1.csv", index=False)
df_test.to_csv(f"{data_dir}/preproc/testPreprocV1.csv", index=False)
