# -*- coding: utf-8 -*-
"""
Model Training
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import os
import sys
import warnings

# Third party imports.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Obtain the current directory path of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils.logger import CustomLogger


# To ignore warnings
warnings.filterwarnings('ignore')


class ModelTraining:
    """
    Model Training Class
    """

    def __init__(self) -> None:
        self.logger = CustomLogger(name='ModelTraining', log_file='model_training.log').get_logger()

        # Define models and pipelines
        self.models = [
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("Adaboost", AdaBoostClassifier(random_state=42)),
            ("ExtraTree", ExtraTreesClassifier(random_state=42)),
            ("BaggingClassifier", BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("DecisionTree", DecisionTreeClassifier(random_state=42)),
            ("NaiveBayes", GaussianNB()),
            ("KNN", KNeighborsClassifier()),
            ("Logistic", LogisticRegression(random_state=42)),
            ("SGD Classifier", SGDClassifier(random_state=42)),
            ("MLPClassifier", MLPClassifier(random_state=42)),
            ("SVM", SVC(random_state=42))
        ]

    def get_training_data(self):
        """
        Get training data

        Returns:
            pd.Dataframe: X train
            pd.Dataframe: X test
            pd.Dataframe: y train
            pd.Dataframe: y test
        """
        # Read processed dataset
        data = pd.read_csv("data/processed/full_dataset.csv")

        # Split data
        class_label = 'Status'
        X = data.drop([class_label], axis=1)
        y = data[class_label]
        X_train, X_test, y_train , y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)

        return X_train, X_test, y_train , y_test

    def compare_classifiers(self, X_train, y_train):
        """
        Compare various classification models and returns their scores from cross validation

        Args:
            X_train (pd.Dataframe): X_train data
            y_train (pd.Dataframe): y_train data

        Returns:
            dict: Model names with their scores from cross validation
        """
        # Define column transformation
        ct = ColumnTransformer([('se', StandardScaler(), ['Price', 'Discount code'])], remainder='passthrough')

        # Create pipelines and evaluate models
        scores = {}
        for name, model in self.models:
            pipeline = Pipeline([('transformer', ct), (name, model)])
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            scores[name] = np.mean(cv_scores)

        return scores

    def get_lead_distribution(self, X_test, y_predicted, y_probabilities):
        """
        Get lead distribution

        Args:
            X_test (pd.Dataframe): X test
            y_predicted (np.ndarray): Y predicted
            y_probabilities (np.ndarray): Y probabilities

        Returns:
            pd.Dataframe: Class predictions and associated probabilities
            pd.Series: Probability distribution
        """
        data_original = pd.read_csv("data/interim/full_dataset.csv")
        X_test_original = data_original.loc[X_test.index]
        # Diccionario de mapeo
        mapping = {0: 'Closed Won', 1: 'Closed Lost', 2: 'Other'}

        # Aplicar el mapeo a y_predicted
        y_predicted_mapped = np.vectorize(mapping.get)(y_predicted)

        # Create a DataFrame containing the class predictions and associated probabilities.
        impact_df = pd.DataFrame({
            'Observation': range(1, len(X_test_original) + 1),
            'Use Case': X_test_original['Use Case'],
            'Discount code': X_test_original['Discount code'],
            'Loss Reason': X_test_original['Loss Reason'],
            'Source': X_test_original['Source'],
            'City': X_test_original['City'],
            'Predicted Class': y_predicted_mapped,
            'Probability Closed-Won': y_probabilities[:, 0],
        })


        # Count the number of observations in each probability range
        probability_bins = pd.cut(impact_df['Probability Closed-Won'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=['25%', '50%', '75%', '100%'])
        probability_distribution = probability_bins.value_counts().sort_index()
        impact_df = impact_df.drop(['Probability Closed-Won'], axis=1)

        return impact_df, probability_distribution

    def run(self):
        """
        Train model

        Returns:
            dict: Training info
        """
        X_train, X_test, y_train , y_test = self.get_training_data()
        model_scores = self.compare_classifiers(X_train, y_train)

        # Show scores
        self.logger.info("Model scores:")
        for model, score in model_scores.items():
            self.logger.info(f"{model}: {score}")

        best_model_name = max(model_scores, key=model_scores.get)
        best_model_index = [name for name, _ in self.models].index(best_model_name)
        best_model = self.models[best_model_index][1]
        self.logger.info(f"Best Model: {best_model_name} with Score: {model_scores[best_model_name]}")

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Probabilities
        y_probabilities = best_model.predict_proba(X_test)
        y_predicted = np.argmax(y_probabilities, axis=1)
        predictions_df, probability_distribution = self.get_lead_distribution(X_test, y_predicted, y_probabilities)

        # Evaluate model performance on the test set
        self.logger.info("Accuracy Score: %s", accuracy_score(y_test, y_pred))
        self.logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

        data = {
            "predictions_df": predictions_df,
            "probability_distribution": probability_distribution,
            "accuracy_score": accuracy_score(y_test, y_pred),
        }

        return data

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run()