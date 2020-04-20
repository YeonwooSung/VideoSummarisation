import pandas as pd
import numpy as np


def iterate_df(df):
    compression_rates = []

    # iterate the dataframe to find min, max, and average of compression rate
    for index, row in df.iterrows():
        original_len = row['original']
        result_len = row['result']

        # calculate the compression rate
        compression_rate = result_len / original_len
        # append the calculated compression rate to the array
        compression_rates.append(compression_rate)

    # calculate the average value of compression rate
    avg_rate = np.average(compression_rates)
    # calculate the min value of compression rate
    min_rate = np.min(compression_rates)
    # calculate the max value of compression rate
    max_rate = np.max(compression_rates)

    print('min={}, max={}, avg={}'.format(min_rate, max_rate, avg_rate))

    return min_rate, max_rate, avg_rate


if __name__ == '__main__':
    df = pd.read_csv('../output/compression.csv')
    epic_train_df = df[df['dataset'] == 'epic_train']
    epic_test_df = df[df['dataset'] == 'epic_test']

    print('EpicKitchens - Training')
    epic_train_min, epic_train_max, epic_train_avg = iterate_df(epic_train_df)
    print('EpicKitchens - Testing')
    epic_test_min, epic_test_max, epic_test_avg = iterate_df(epic_test_df)

    #TODO
