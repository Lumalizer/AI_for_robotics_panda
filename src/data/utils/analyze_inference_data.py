# take all folders in the inference_data folder and analyze the data

import os
import pandas as pd
import pprint

root = "datasets/inference_data"
subproject = "determine_octo_parameters"
folders = os.listdir(os.path.join(root, subproject))

csvs = []
for folder in folders:
    csvs.append(os.path.join(root, subproject, folder, "data.csv"))
    
dfs = []
for csv in csvs:
    df = pd.read_csv(csv)
    dfs.append(df)
    
df = pd.concat(dfs)

df.to_csv("datasets/combined_data.csv", index=False)

print(df.head())
print(df.tail())

# remove episode, mode, model_type, window_size, proprio, and date columns
df = df.drop(columns=["mode", "model_type", "window_size", "proprio", "date"])


# group by task desciption, prediction_horizon, and execution_horizon
# sum the success_or_failre column
# average the total_time column
# count the number of episodes

grouped = df.groupby(["task_description", "prediction_horizon", "execution_horizon"]).agg(
    success_or_failure = ("success_or_failure", "mean"),
    total_time = ("total_time", "mean"),
    episodes = ("episode", "count")
)

print(grouped)
