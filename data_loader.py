"""
This module is used to merge the data files into a single dataframe.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def merge_data():
    # Load the Current Population Survey Data
    cps_data = pd.read_stata("cps_data.dta")
    to_drop = cps_data[cps_data["pernum"] != 1].index
    cps_data = cps_data.drop(to_drop)
    cps_data['year'] = cps_data['year'].astype(int)

    # Collapse the CPS data, saving average income, average property tax, and population for each year, state, and race
    collapsed = cps_data.groupby(['year', 'statefip', 'race']).apply(
        lambda x: pd.Series({
            'avg_income': (x['hhincome'] * x['asecwt']).sum() / x['asecwt'].sum(),
            'population': x['asecwt'].sum(),
            'avg_prop_tax': (x['proptax'] * x['asecwt']).sum() / x['asecwt'].sum()
        })
    ).reset_index().dropna()

    # Create averages for the state in each year...
    # Pivot the collapsed dataframe to create race-specific columns for population
    collapsed_pivot = collapsed.pivot_table(
        index=['year', 'statefip'],
        columns='race',
        values=['population', 'avg_income', 'avg_prop_tax'],
        aggfunc='sum'
    )

    # Flatten the multi-level columns
    collapsed_pivot.columns = ['_'.join(map(str, col)).strip() for col in collapsed_pivot.columns]

    # Calculate total population and proportions for each race
    collapsed_pivot['total_population'] = collapsed_pivot.filter(like='population_').sum(axis=1)
    for race in collapsed['race'].unique():
        collapsed_pivot[f'proportion_{race}'] = (
            collapsed_pivot[f'population_{race}'] / collapsed_pivot['total_population']
        )

    # Multiply avg_prop_tax columns by their corresponding proportions
    races = collapsed['race'].unique()

    weighted_avg_prop_tax = []
    for race in races:
        weighted_avg_prop_tax.append(
            collapsed_pivot[f'avg_prop_tax_{race}'] * collapsed_pivot[f'proportion_{race}']
        )

    # Sum the weighted values to calculate the weighted average
    collapsed_pivot['avg_prop_tax'] = sum(weighted_avg_prop_tax)

    # Similarly, calculate weighted avg_income
    weighted_avg_income = []
    for race in races:
        weighted_avg_income.append(
            collapsed_pivot[f'avg_income_{race}'] * collapsed_pivot[f'proportion_{race}']
        )

    collapsed_pivot['avg_income'] = sum(weighted_avg_income)

    # Keep only the required columns
    columns_to_keep = ['total_population', 'avg_income', 'avg_prop_tax'] + \
                    [col for col in collapsed_pivot.columns if 'proportion' in col]
    collapsed_pivot = collapsed_pivot[columns_to_keep]

    cols_to_keep = (collapsed_pivot != 0).sum().sort_values(ascending=False).index[:7].sort_values()

    # Reset index to return to a flat DataFrame
    collapsed = collapsed_pivot[cols_to_keep].reset_index()
    # ...Finished averaging the states in each year


    # Load the housing data
    price_by_state = pd.read_csv("Data_Files/price_by_state_cleaned.csv")

    price_by_state.columns = np.concatenate([['Date'], price_by_state.columns[1:].values])
    price_by_state['Date'] = pd.to_datetime(price_by_state['Date'])
    price_by_state.set_index('Date', inplace=True)

    price_by_state = price_by_state.reset_index().melt(id_vars='Date', var_name='State', value_name='Median Home Price')

    # Use the dates from the housing data to interpolate the CPS data
    # Create a new DataFrame with monthly dates for interpolation
    date_range = price_by_state['Date'].drop_duplicates()

    # Expand the date range for each state and race combination
    expanded_index = pd.MultiIndex.from_product(
        [date_range, collapsed['statefip']],
        names=['Date', 'statefip']
    )
    expanded_df = pd.DataFrame(index=expanded_index).reset_index()

    # Merge with collapsed to align data
    expanded_df['year'] = expanded_df['Date'].dt.year
    merged_df = expanded_df.merge(collapsed, on=['year', 'statefip'], how='left')

    # Drop the 'year' column and fill missing values forward within each state and race
    merged_df.drop(columns=['year'], inplace=True)
    merged_df.sort_values(by=['statefip', 'Date'], inplace=True)
    merged_df.fillna(method='ffill', inplace=True)

    # Final interpolated DataFrame
    interpolated_collapsed = merged_df.drop_duplicates()

    # Merge the housing data with the CPS data
    merged_data = pd.merge(interpolated_collapsed, price_by_state, left_on=['Date', 'statefip'], right_on=['Date', 'State']).drop(columns=['State'])

    # Create multiindex array
    merged_data_multiindex = merged_data.set_index(['statefip', 'Date']).sort_index()

    # Interpolate the data to fill in missing months for demographic data
    interpolated_df = []

    for group, df in merged_data_multiindex[merged_data_multiindex.columns[:-1]].groupby('statefip'):
        df_len = df.shape[0]
        n_months = df_len // 12 + 1
        interp1d_func = interp1d(np.linspace(0, df_len, n_months), df.iloc[::12].values, axis=0, fill_value="extrapolate")
        interp_df = pd.DataFrame(interp1d_func(np.arange(df_len)), columns=df.columns, index = df.index)
        interpolated_df.append(interp_df)

    interpolated_df
    interpolated_df = pd.concat(interpolated_df)
    interpolated_df['Median Home Price'] = merged_data_multiindex['Median Home Price'].values
    interpolated_df

    # Save the merged data to a csv file
    interpolated_df.to_csv("Data_Files/state_full.csv")


def load_data():
    try:
        # Load the merged data
        merged_data = pd.read_csv("Data_Files/state_full.csv", index_col=[0, 1])

    except FileNotFoundError:
        # If 
        merge_data()
        merged_data = pd.read_csv("Data_Files/state_full.csv", index_col=[0, 1])

    # Convert the index to DatetimeIndex
    merged_data.index = pd.MultiIndex.from_tuples(
        [(statefip, pd.to_datetime(date)) for statefip, date in merged_data.index],
        names=['statefip', 'Date']
    )

    return merged_data