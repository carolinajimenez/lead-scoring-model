# -*- coding: utf-8 -*-
"""
Data Preparation - 1. Data Cleaning
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.


# Third party imports.
from numpy import percentile
import pandas as pd
import seaborn as sns
import plotly.express as px


class DataCleaning:
    """
    Data cleaning involves fixing systematic problems or errors in messy data.
    """

    def __init__(self, main_path) -> None:
        """
        Constructor.

        Args:
            * main_path (str): Main Dataset Path.
        """
        self.main_path = main_path

    def load_dataset(self, dataset_path):
        """
        Load dataset and save them as dataframes.

        Args:
            * dataset_path (str): Dataset Path.

        Returns:
            * (pd.Dataframe): Pandas Dataframe.
        """
        dataframe = pd.read_csv(f"{self.main_path}/{dataset_path}")
        return dataframe

    def inspect_dataset(self, dataframe):
        """
        Inspect information from dataset

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.
        """
        print("\n* Shape *", dataframe.shape, end="\n\n")
        print("* Dataframe visualization *\n")
        print(dataframe.head(), end="\n\n")
        print("* Dataframe overview *\n")
        dataframe.info()
        print("\n* Number of variables of each data type *\n")
        print(dataframe.dtypes.value_counts(), end="\n\n")
        print("* Null values *\n")
        print("- - Are there null values?\n")
        print(dataframe.isnull().any(), end="\n\n")
        print("- - If we have observations with null values, how many do we have for each variable?\n")
        print(dataframe.isnull().sum().sort_values(ascending=False), end="\n\n")
        print("- - How many null values do we have in total in the data set?\n")
        print(dataframe.isnull().sum().sum(), end="\n\n")
        print("- - What is the proportion of null values for each variable?\n")
        self.visualize_proportion_null_values(dataframe)
        print("\n\n")

    def visualize_proportion_null_values(self, dataframe):
        """
        Visualize proportion of null values for each variable

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.
        """
        (
            dataframe
            .isnull()
            .melt(value_name='missing')
            .pipe(
                lambda df: (
                    sns.displot(
                        data=df,
                        y='variable',
                        hue='missing',
                        multiple='fill',
                        aspect=2
                    )
                )
            )
        )

    def identify_columns_with_single_value(self, dataframe):
        """
        Identify columns that contain a single value.
        Columns that have a single observation or value are probably useless for modeling.
        These columns or predictors are referred to zero-variance predictors as if we measured
        the variance (average value from the mean), it would be zero.

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.

        Returns:
            (pd.Series): Number of unique values for each column.
        """
        print("\nColumn index - Number of unique values for each column.")
        # Get number of unique values for each column
        counts = dataframe.nunique()
        print(counts, end="\n\n")
        return counts

    def delete_columns_with_single_value(self, dataframe, counts):
        """
        Delete columns that contain a single value

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.
            * counts (pd.Series): Number of unique values for each column.

        Returns:
            * (pd.Dataframe): Pandas Dataframe without columns with a single value.
        """
        # Record columns to delete
        to_del = [i for i,v in enumerate(counts) if v == 1]
        # Drop useless columns
        print("\nBefore removing columns - Shape:", dataframe.shape, end="\n\n")
        dataframe.drop(to_del, axis=1, inplace=True)
        print("After removing columns - Shape:", dataframe.shape, end="\n\n")
        return dataframe

    def identify_row_with_duplicate_data(self, dataframe):
        """
        Identify rows that contain duplicate data.
        Rows that have identical data are could be useless to the modeling process,
        if not dangerously misleading during model evaluation.

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.
        """
        # Calculate duplicates
        dups = dataframe.duplicated()
        # Report if there are any duplicates
        print("\nAre there duplicates?:", dups.any(), end="\n\n")
        # List all duplicate rows
        print(dataframe[dups])

    def delete_row_with_duplicate_data(self, dataframe):
        """
        Delete columns that contain a single value

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe.

        Returns:
            * (pd.Dataframe): Pandas Dataframe without rows that contain duplicate data.
        """
        # Delete duplicate rows
        print("\nBefore removing rows - Shape:", dataframe.shape, end="\n\n")
        dataframe.drop_duplicates(inplace=True)
        print("After removing rows - Shape:", dataframe.shape, end="\n\n")
        return dataframe

    def drop_columns(self, dataframe, columns_to_drop):
        """
        Function to remove specific columns from a DataFrame.

        Parameters:
            * dataframe (pd.Dataframe): Pandas Dataframe.
            * columns_to_drop (list): Column names to drop.

        Returns:
            * (pd.Dataframe): DataFrame with the specified columns removed.
        """
        df_copy = dataframe.drop(columns=columns_to_drop, errors='ignore')
        return df_copy

    def handle_missing_values(self, dataframe, columns, method='drop'):
        """
        Function to handle null values in a DataFrame.
        Having missing values in a dataset can cause errors with some machine learning algorithms.

        Parameters:
            * dataframe (pd.Dataframe): Pandas Dataframe.
            * columns (list): Column names or single column to segment.
            * method (str): Method to handle null values. Options: 'drop' to remove rows with null values,
                    'mean' to replace null values with column mean, 'median' to replace null values with
                    column median values with median of column, 'mode' to replace the null values with
                    mode of column. Default, 'drop'.

        Returns:
            * (pd.Dataframe): DataFrame with the null values handled according to the specified method.
        """
        df_copy = dataframe.copy()
        if method == 'drop':
            df_copy = df_copy.dropna(subset=columns)
        elif method == 'mean':
            df_copy[columns] = df_copy[columns].fillna(df_copy[columns].mean())
        elif method == 'median':
            df_copy[columns] = df_copy[columns].fillna(df_copy[columns].median())
        elif method == 'mode':
            df_copy[columns] = df_copy[columns].fillna(df_copy[columns].mode().iloc[0])
        else:
            raise ValueError("Invalid method. Valid options: 'drop', 'mean', 'median', 'mode'.")
        return df_copy

    def impute_nulls_by_group_using_mode(self, dataframe, group_columns, target_column):
        """
        Imputes null values in a specific column of a DataFrame using mode of each group
        formed by the unique combinations of values in the grouping columns.

        Parameters:
            * dataframe (pd.Dataframe): Pandas Dataframe.
            * group_columns (list): Column names to group.
            * target_column (str): Name of the column to transform (impute null values).

        Return:
            * (pd.Dataframe): DataFrame with the null values imputed in the specified column.
        """
        df_copy = dataframe.copy()
        # Imputación basada en la mediana de la columna target_column para cada combinación única de valores en group_columns
        df_copy[target_column] = df_copy.groupby(group_columns)[target_column].transform(lambda x: x.fillna(x.mode().iloc[0]))
        return df_copy

    def impute_nulls_by_group_using_median(dataframe, group_columns, target_column):
        """
        Imputes null values in a specific column of a DataFrame using median of each group
        formed by the unique combinations of values in the grouping columns.

        Parameters:
            * dataframe (pd.Dataframe): Pandas Dataframe.
            * group_columns (list): Column names to group.
            * target_column (str): Name of the column to transform (impute null values).

        Return:
            * (pd.Dataframe): DataFrame with the null values imputed in the specified column.
        """
        df_copy = dataframe.copy()
        # Imputación basada en la mediana de la columna target_column para cada combinación única de valores en group_columns
        df_copy[target_column] = df_copy.groupby(group_columns)[target_column].transform(lambda x: x.fillna(x.median()))
        return df_copy

    def view_data_distribution(self, dataframe):
        fig = px.histogram(dataframe)
        fig.show()

    def identify_outliers(self, dataframe):
        # calculate interquartile range
        q25 = dataframe.quantile(0.25)
        q75 = dataframe.quantile(0.75)
        iqr = q75 - q25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = dataframe[((dataframe<(lower)) | (dataframe>(upper)))]
        print("\nIdentified outliers: %d" % outliers)
        return outliers

    def remove_outliers(self, dataframe, lower, upper):
        outliers_removed = [x for x in dataframe if x >= lower and x <= upper]
        print("\nNon-outlier observations: %d" % len(outliers_removed))


def main():
    data_cleaning = DataCleaning(main_path="data/raw")

    # LEADS
    leads_df = data_cleaning.load_dataset(dataset_path="leads.csv")

    data_cleaning.inspect_dataset(leads_df)

    # counts = data_cleaning.identify_columns_with_single_value(leads_df)
    # leads_df = data_cleaning.delete_columns_with_single_value(leads_df, counts)

    # data_cleaning.identify_row_with_duplicate_data(leads_df)
    # leads_df = data_cleaning.delete_row_with_duplicate_data(leads_df)

    # data_cleaning.view_data_distribution(leads_df)
    # outliers = data_cleaning.identify_outliers(leads_df)

    # OFFERS
    offers_df = data_cleaning.load_dataset(dataset_path="offers.csv")

    data_cleaning.inspect_dataset(offers_df)

    # counts = data_cleaning.identify_columns_with_single_value(offers_df)
    # offers_df = data_cleaning.delete_columns_with_single_value(offers_df, counts)

    # data_cleaning.identify_row_with_duplicate_data(offers_df)
    # offers_df = data_cleaning.delete_row_with_duplicate_data(offers_df)

    # data_cleaning.view_data_distribution(offers_df)
    # outliers = data_cleaning.identify_outliers(offers_df)


if __name__ == "__main__":
    main()
