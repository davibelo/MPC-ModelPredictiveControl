import pandas as pd
import matplotlib.pyplot as plt

def get_unit_step_responses(df_u, df_y, step_count, response_size):
    df_responses = pd.DataFrame()

    # Iterate over the columns of df_u
    for column in df_u.columns:
        # Initialize variables to store the step moment and previous u value
        step_moment = None
        prev_u = None
        step_counter = 0

        # Iterate over the rows of df_u
        for index, row in df_u.iterrows():
            # If the previous u value is not None and the current u value is different from the previous one
            if prev_u is not None and row[column] != prev_u:
                step_counter += 1  # Increment the step counter
                if step_counter == step_count:  # Check if this is the second step moment
                    step_moment = index  # Set the step moment to the current index
                    step_amplitude = row[column] - prev_u  # Calculate the step amplitude
                    break  # Break the loop as we found the second step moment

            prev_u = row[column]  # Update the previous u value

        # Ensure that the step moment allows enough data for the response size
        if step_moment is not None and step_moment + response_size <= df_u.index[-1]:
            # Extract the output response (from the step moment to RESPONSE_SIZE rows after)
            step_responses = df_y.loc[step_moment:step_moment + response_size - 1]

            # Reset the index of the response DataFrame
            step_responses.reset_index(drop=True, inplace=True)

            # Rename the columns of the response DataFrame
            step_responses.columns = [f"{column} x {col}" for col in step_responses.columns]

            # Divide each value in the response column by the amplitude of the step
            step_responses = step_responses / step_amplitude

            # Subtract all values in the column by the first value
            step_responses = step_responses - step_responses.iloc[0]

            # Concatenate the response DataFrame to the df_responses DataFrame
            df_responses = pd.concat([df_responses, step_responses], axis=1)

    return df_responses

def plot_and_save(dataframe, figsize, xticks_increment, y_num_bins, fig_name, dataframe2=None):
    # Number of columns to plot
    num_columns = len(dataframe.columns)

    # Creating subplots
    fig, axs = plt.subplots(num_columns, 1, figsize=figsize)

    # Plotting each column on a separate subplot
    for i, column in enumerate(dataframe.columns):
        axs[i].plot(dataframe.index, dataframe[column], label='unit step 1')
        axs[i].set_title(column)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Values')
        axs[i].set_xticks(range(0, len(dataframe.index), xticks_increment))
        axs[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees
        axs[i].grid(True)  # Add gridlines
        axs[i].locator_params(axis='y', nbins=y_num_bins)

        # If a second DataFrame is provided, plot it on the same subplot
        if dataframe2 is not None:
            axs[i].plot(dataframe2.index, dataframe2[column], label='unit step 2')

    plt.tight_layout()  # Adjust layout to not overlap subplots
    axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), framealpha=0.25)
    plt.savefig(f'figures/{fig_name}.png')