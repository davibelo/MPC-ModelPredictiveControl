import pandas as pd
import matplotlib.pyplot as plt
from src.StepCalc import *
import joblib

NUMBER_OF_INPUTS = 2
RESPONSE_SIZE = 60

# Step 1: Read the CSV file
df = pd.read_csv('LPGdesbutanizerStepTest.csv', delimiter=";", index_col=0)
df.columns = ['TBOTTOM', 'QREFLUX', 'C2LPG', 'C5LPG']
print('test data:')
print(df)

# Split the data into input and output DataFrames
df_u = df.iloc[:, :NUMBER_OF_INPUTS]
df_y = df.iloc[:, NUMBER_OF_INPUTS:]

plot_and_save(dataframe1=df,
              figsize=(10, 10),
              xticks_increment=25,
              y_num_bins=5,
              fig_name='LPGdebutanizerStepTest',
              label1='step tests')

# Get first step responses
df_responses1 = get_unit_step_responses(df_u, df_y, 1, RESPONSE_SIZE)

# Get second step responses
df_responses2 = get_unit_step_responses(df_u, df_y, 2, RESPONSE_SIZE)

plot_and_save(dataframe1=df_responses1,
              dataframe2=df_responses2,
              figsize=(8, 6),
              xticks_increment=5,
              y_num_bins=5,
              fig_name='LPGdebutanizerUnitSteps',
              label1='unit step 1',
              label2='unit step 2')

G11 = df_responses1['C2LPG x TBOTTOM'].to_numpy()
G12 = df_responses1['C2LPG x QREFLUX'].to_numpy()
G21 = df_responses1['C5LPG x TBOTTOM'].to_numpy()
G22 = df_responses1['C5LPG x QREFLUX'].to_numpy()

Gstep = [[G11, G12], [G21, G22]]
joblib.dump(Gstep, 'outputs/Gstep.joblib')