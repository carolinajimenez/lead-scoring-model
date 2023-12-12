# -*- coding: utf-8 -*-
"""
Auxiliary functions
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
from typing import Dict
import warnings

# Third party imports.
import numpy as np
import pandas as pd
import shimoku_api_python as shimoku


# To ignore warnings
warnings.filterwarnings('ignore')


#--------------------AUXILIARY FUNCTIONS--------------------#
def get_label_columns(table_data: pd.DataFrame) -> Dict:
    print(table_data.head())
    return {
        'Predicted Class': {
            ("Closed Won"): '#20C69E',  #  Green
            ("Closed Lost"): '#F86C7D',  #  Red
            ("Other"): '#F2BB67',  #  Yellow
        },
    }


#--------------------DASHBOARD FUNCTIONS--------------------#
def page_header(shimoku_client: shimoku.Client, order: int):
    prediction_header = (
        "<head>"
        "<style>"  # Styles title
        ".component-title{height:auto; width:100%; "
        "border-radius:16px; padding:16px;"
        "display:flex; align-items:center;"
        "background-color:var(--chart-C1); color:var(--color-white);}"
        "</style>"
        # Start icons style
        "<style>.big-icon-banner"
        "{width:48px; height: 48px; display: flex;"
        "margin-right: 16px;"
        "justify-content: center;"
        "align-items: center;"
        "background-size: contain;"
        "background-position: center;"
        "background-repeat: no-repeat;"
        "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
        "</style>"
        # End icons style
        "<style>.base-white{color:var(--color-white);}</style>"
        "</head>"  # Styles subtitle
        "<div class='component-title'>"
        "<div class='big-icon-banner'></div>"
        "<div class='text-block'>"
        "<h1>Predictions</h1>"
        "<p class='base-white'>"
        "Lead scoring prediction</p>"
        "</div>"
        "</div>"
    )
    shimoku_client.plt.html(html=prediction_header, order=order)

def prediction_table(shimoku_client: shimoku.Client, order: int, binary_prediction_table: pd.DataFrame):
    prediction_table_header = (
        '<div style="width:100%; height:90px; "><h4>Lead predicton & factors</h4>'
        '<p>Affectation values for each lead</p></div>'
    )
    shimoku_client.plt.html(html=prediction_table_header, order=order)

    label_columns = get_label_columns(binary_prediction_table)

    shimoku_client.plt.table(
        order=order+1, data=binary_prediction_table,
        label_columns=label_columns,
        columns_options={
            'Observation': {'width': 182},
            'Predicted Class': {'width': 203},
            'Use Case': {'width': 178},
            'Discount code': {'width': 206},
            'Loss Reason': {'width': 199},
            'Source': {'width': 159},
            'City': {'width': 154},
        }
    )

    table_explanaiton = (
        "<head>"
        "<style>.banner"
        "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
        "background-size: cover;"
        "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
        "color:var(--color-white);}"
        "</style>"
        "</head>"
        "<a href='https://shimoku.webflow.io/product/churn-prediction' target='_blank'>"  # link
        "<div class='banner'>"
        "<p class='base-white'>"
        "This table containing the class predictions and associated probabilities."
        "</p>"
        "</div>"
        "</a>"
    )
    shimoku_client.plt.html(html=table_explanaiton, order=order+2)

def distribution_chart(shimoku_client: shimoku.Client, order: int, doughnut_chart_data: Dict):
    shimoku_client.plt.free_echarts(raw_options=doughnut_chart_data, order=order, cols_size=5, rows_size=2)
