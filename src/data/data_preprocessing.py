# -*- coding: utf-8 -*-
"""
Data Pre-Processing
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import os

# Third party imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
from sklearn.preprocessing import LabelEncoder

os.chdir('..')

# *** Reading dataset ***

leads_data = pd.read_csv("data/raw/leads.csv")
offers_data = pd.read_csv("data/raw/offers.csv")

# *** Feature Engineering ***

# Delete rows with null values in the 'Id' column of the Leads dataset
leads_data_cleaned = leads_data.dropna(subset=['Id'])

# Drop multiple columns
# 'First Name' is irrelevant
# The columns of Leads dataset such as 'Use Case' and 'Created Date' are the same as the columns of Offers dataset
# The 'Status' and 'Converted' columns of Leads dataset refers to the 'Status' column of Offers dataset
leads_data_cleaned = leads_data_cleaned.drop(['First Name', 'Use Case', 'Created Date', 'Status', 'Converted'], axis = 1)

leads_data_cleaned.to_csv("data/interim/leads_data_cleaned.csv", index=False)

# Merge the datasets using the 'Id' column as a key
full_dataset = pd.merge(offers_data, leads_data_cleaned, on='Id', how='left')
full_dataset.to_csv("data/interim/full_dataset.csv", index=False)

# Drop multiple columns
# 'Id' is irrelevant
# 'Discarded/Nurturing Reason' and 'Acquisition Campaign' have more than 80% null values.
# The missing data is significant and may not provide valuable information for the model.
full_dataset_preprocessed = full_dataset.drop(['Id', 'Discarded/Nurturing Reason', 'Acquisition Campaign'], axis = 1)

full_dataset_preprocessed['Created Date'] = pd.to_datetime(full_dataset_preprocessed['Created Date'], format="%Y-%m-%d")
full_dataset_preprocessed['Close Date'] = pd.to_datetime(full_dataset_preprocessed['Close Date'], format="%Y-%m-%d")

# Fill null values in 'Loss Reason'
# If 'Status' is 'Closed Lost', fill in 'no response'
full_dataset_preprocessed['Loss Reason'] = np.where((full_dataset_preprocessed['Status'] == 'Closed Lost') & (full_dataset_preprocessed['Loss Reason'].isnull()), 'no response', full_dataset_preprocessed['Loss Reason'])

# If 'Status' is 'Closed Won', fill with 'Loss Reason' mode fora 'Closed Won'
mode_closed_won = full_dataset_preprocessed.loc[full_dataset_preprocessed['Status'] == 'Closed Won', 'Loss Reason'].mode()[0]
full_dataset_preprocessed['Loss Reason'].fillna(mode_closed_won, inplace=True)

for col in full_dataset_preprocessed.columns:
    if full_dataset_preprocessed[col].dtype in ['object', 'datetime64[ns]']:
        full_dataset_preprocessed[col] = full_dataset_preprocessed[col].fillna(full_dataset_preprocessed[col].mode()[0])
    elif full_dataset_preprocessed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        full_dataset_preprocessed[col] = full_dataset_preprocessed[col].fillna(full_dataset_preprocessed[col].mean())

full_dataset_preprocessed['Created Year']= full_dataset_preprocessed['Created Date'].dt.year
full_dataset_preprocessed['Created Month']= full_dataset_preprocessed['Created Date'].dt.month
full_dataset_preprocessed['Close Year']= full_dataset_preprocessed['Close Date'].dt.year
full_dataset_preprocessed['Close Month']= full_dataset_preprocessed['Close Date'].dt.month

full_dataset_preprocessed = full_dataset_preprocessed.drop(['Created Date', 'Close Date'], axis = 1)

clase_mapping = {'Closed Won': 'Closed Won', 'Closed Lost': 'Closed Lost'}
# Assign 'Other' to all classes that are not 'Closed Won' or 'Closed Lost'
full_dataset_preprocessed['Status'] = full_dataset_preprocessed['Status'].map(clase_mapping).fillna('Other')

# Creating a instance of label Encoder.
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_columns = full_dataset_preprocessed.select_dtypes(['object', 'datetime64[ns]']).columns
categorical_columns = list(set(categorical_columns))

# Aplicar LabelEncoder a cada columna categórica
for column in categorical_columns:
    if column in full_dataset_preprocessed.columns:
        full_dataset_preprocessed[column] = label_encoder.fit_transform(full_dataset_preprocessed[column])

full_dataset_preprocessed.to_csv("data/processed/full_dataset.csv", index=False)
