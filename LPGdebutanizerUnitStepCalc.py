import pandas as pd
import matplotlib.pyplot as plt
from src.StepCalc import *

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

plot_and_save(dataframe=df,
              figsize=(10, 10),
              xticks_increment=25,
              y_num_bins=5,
              fig_name='LPGdebutanizerStepTest')

# Get first step responses
df_responses1 = get_unit_step_responses(df_u, df_y, 1, RESPONSE_SIZE)
plot_and_save(dataframe=df_responses1,
              figsize=(8, 6),
              xticks_increment=5,
              y_num_bins=5,
              fig_name='LPGdebutanizerUnitStep1')

# Get second step responses
df_responses2 = get_unit_step_responses(df_u, df_y, 2, RESPONSE_SIZE)
plot_and_save(dataframe=df_responses2,
              figsize=(8, 6),
              xticks_increment=5,
              y_num_bins=5,
              fig_name='LPGdebutanizerUnitStep2')

plot_and_save(dataframe=df_responses1,
              dataframe2=df_responses2,
              figsize=(8, 6),
              xticks_increment=5,
              y_num_bins=5,
              fig_name='LPGdebutanizerUnitSteps')
