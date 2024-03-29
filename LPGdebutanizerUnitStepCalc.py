import pandas as pd
import matplotlib.pyplot as plt
from src.StepCalc import *

# Step 1: Read the CSV file
df = pd.read_csv('LPGdesbutanizerStepTest.csv', delimiter=";", index_col=0)
df.columns = ['TBOTTOM', 'QREFLUX', 'C2LPG', 'C5LPG']
print('test data:')
print(df)

# Number of columns to plot
num_columns = len(df.columns)

# Creating subplots
fig, axs = plt.subplots(num_columns, 1, figsize=(10, 10))  # Adjust figsize as needed

# Plotting each column on a separate subplot
for i, column in enumerate(df.columns):
    axs[i].plot(df.index, df[column])
    axs[i].set_title(column)
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Values')
    axs[i].set_xticks(range(0, len(df.index), 25))
    axs[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees
    axs[i].grid(True)  # Add gridlines
    axs[i].locator_params(axis='y', nbins=5)

plt.tight_layout()  # Adjust layout to not overlap subplots
FIG_NAME = 'LPGdebutanizerStepTest'
plt.savefig(f'figures/{FIG_NAME}.png')

NUM_U = 2
df_u = df.iloc[:, :NUM_U]
df_y = df.iloc[:, NUM_U:]

# GET FIRST STEP RESPONSES
# Constant for response size
RESPONSE_SIZE = 60

# Call the function to calculate responses
df_responses1 = calculate_responses(df_u, df_y, 1, RESPONSE_SIZE)

# Compute the maximum absolute value of each column in df_responses
max_abs_values = df_responses1.abs().max()

# Convert the maximum absolute values to scientific notation with two decimal places
max_abs_values_scientific = max_abs_values.apply(lambda x: '{:.2e}'.format(x))

# Print the maximum absolute value of each column in scientific notation
print('max abs values on first unit step responses:')
print(max_abs_values_scientific)

# Number of columns to plot
num_columns = len(df_responses1.columns)

# Creating subplots
fig, axs = plt.subplots(num_columns, 1, figsize=(8, 6))  # Adjust figsize as needed

# Plotting each column on a separate subplot
for i, column in enumerate(df_responses1.columns):
    axs[i].plot(df_responses1.index, df_responses1[column])
    axs[i].set_title(column)
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Values')
    axs[i].set_xticks(range(0, len(df_responses1.index), 5))
    axs[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees
    axs[i].grid(True)  # Add gridlines

plt.tight_layout()  # Adjust layout to not overlap subplots
FIG_NAME = 'LPGdebutanizerUnitStep1'
plt.savefig(f'figures/{FIG_NAME}.png')

# GET SECOND STEP RESPONSES
# Constant for response size
RESPONSE_SIZE = 60

# Call the function to calculate responses
df_responses2 = calculate_responses(df_u, df_y, 2, RESPONSE_SIZE)

# Compute the maximum absolute value of each column in df_responses
max_abs_values = df_responses2.abs().max()

# Convert the maximum absolute values to scientific notation with two decimal places
max_abs_values_scientific = max_abs_values.apply(lambda x: '{:.2e}'.format(x))

# Print the maximum absolute value of each column in scientific notation
print('max abs values on second unit step responses:')
print(max_abs_values_scientific)

# Number of columns to plot
num_columns = len(df_responses2.columns)

# Creating subplots
fig, axs = plt.subplots(num_columns, 1, figsize=(8, 6))  # Adjust figsize as needed

# Plotting each column on a separate subplot
for i, column in enumerate(df_responses2.columns):
    axs[i].plot(df_responses2.index, df_responses2[column])
    axs[i].set_title(column)
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Values')
    axs[i].set_xticks(range(0, len(df_responses2.index), 5))
    axs[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees
    axs[i].grid(True)  # Add gridlines

plt.tight_layout()  # Adjust layout to not overlap subplots
FIG_NAME = 'LPGdebutanizerUnitStep2'
plt.savefig(f'figures/{FIG_NAME}.png')
