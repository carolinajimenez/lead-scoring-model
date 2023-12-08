"""
Lead Scoring Model
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import warnings

# Third party imports.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import CustomLogger

# To ignore warnings
warnings.filterwarnings('ignore')

class DataPreprocessing:
    def __init__(self) -> None:
        self.logger = CustomLogger(name='DataPreprocessing', log_file='reports/logs_data_preprocessing.txt')

    def read_dataset(self, path):
        data = pd.read_csv(path)
        return data

    def analyze_data(self, data):
        self.logger.log_info("Top 5 observations of the dataset")
        print(data.head())
        self.logger.log_info("Data type and information about data")
        print(data.info())
        self.logger.log_info("Check for duplication")
        print(data.nunique())
        self.logger.log_info("Missing values - Real values")
        print(data.isnull().sum())
        self.logger.log_info("Missing values - Percentages")
        print((data.isnull().sum()/(len(data)))*100)


data_preprocessing = DataPreprocessing()
data = data_preprocessing.read_dataset("data/raw/leads.csv")
data_preprocessing.analyze_data(data)

