# -*- coding: utf-8 -*-
"""
APP
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import os
import sys
import warnings

# Third party imports.
import pandas as pd
import shimoku_api_python as shimoku

# Obtain the current directory path of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.train_model import ModelTraining
from src.utils.aux import page_header, prediction_table, distribution_chart


# To ignore warnings
warnings.filterwarnings('ignore')


def main():
    # Get the data dictionary
    trainer = ModelTraining()
    data = trainer.run()

    # Client initialization
    access_token = os.getenv('SHIMOKU_TOKEN')
    universe_id: str = os.getenv('SHIMOKU_UNIVERSE_ID')
    workspace_id: str = os.getenv('SHIMOKU_WORKSPACE_ID')

    s = shimoku.Client(
        access_token=access_token,
        universe_id=universe_id,
        async_execution=True,
        verbosity='INFO',
    )
    s.set_workspace(workspace_id)
    s.set_board('Lead Scoring Board')
    s.set_menu_path('Lead Scoring')

    # Create dashboard tasks
    page_header(s, 0)
    prediction_table(s, 7, data['predictions_df'])
    # distribution_chart(s, 11, data['doughnut_chart_data'])

    # Execute all tasks
    s.run()

if __name__ == '__main__':
    main()
