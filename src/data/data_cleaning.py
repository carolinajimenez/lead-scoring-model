# -*- coding: utf-8 -*-
"""
Data Preparation - 1. Data Cleaning
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.


# Third party imports.
import pandas as pd


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
        print("\nShape:", dataframe.shape, end="\n\n")
        print(dataframe.head(), end="\n\n")
        print(dataframe.info(), end="\n\n")

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
        print(counts)
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
        dataframe.drop(to_del, axis=1, inplace=True)
        print("\nAfter removing columns - Shape:", dataframe.shape, end="\n\n")
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
        print("\nAre there duplicates?:", dups.any())
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
        dataframe.drop_duplicates(inplace=True)
        print("\nAfter removing rows - Shape:", dataframe.shape, end="\n\n")
        return dataframe


def main():
    data_cleaning = DataCleaning(main_path="data/raw")

    leads_df = data_cleaning.load_dataset(dataset_path="leads.csv")
    data_cleaning.inspect_dataset(leads_df)
    counts = data_cleaning.identify_columns_with_single_value(leads_df)
    leads_df = data_cleaning.delete_columns_with_single_value(leads_df, counts)
    data_cleaning.identify_row_with_duplicate_data(leads_df)
    leads_df = data_cleaning.delete_row_with_duplicate_data(leads_df)

    offers_df = data_cleaning.load_dataset(dataset_path="offers.csv")
    data_cleaning.inspect_dataset(offers_df)
    counts = data_cleaning.identify_columns_with_single_value(offers_df)
    offers_df = data_cleaning.delete_columns_with_single_value(offers_df, counts)
    data_cleaning.identify_row_with_duplicate_data(offers_df)
    offers_df = data_cleaning.delete_row_with_duplicate_data(offers_df)


if __name__ == "__main__":
    main()
