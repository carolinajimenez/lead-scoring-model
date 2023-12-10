"""
Model Training
"""
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

        # Define models and pipelines
        self.models = [
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("Adaboost", AdaBoostClassifier(random_state=42)),
            ("ExtraTree", ExtraTreesClassifier(random_state=42)),
            ("BaggingClassifier", BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("DecisionTree", DecisionTreeClassifier(random_state=42)),
            ("KNN", KNeighborsClassifier()),
            ("Logistic", LogisticRegression(random_state=42)),
            ("SGD Classifier", SGDClassifier(random_state=42)),
            ("MLPClassifier", MLPClassifier(random_state=42)),
            ("NaiveBayes", GaussianNB()),
            ("SVM", SVC(random_state=42))
        ]

        # Create pipelines and evaluate models
        scores = {}
        for name, model in self.models:
            pipeline = Pipeline([('transformer', ct), (name, model)])
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            scores[name] = np.mean(cv_scores)

        return scores

    def run(self):
        X_train, X_test, y_train , y_test = self.get_training_data()
        model_scores = self.compare_classifiers(X_train, y_train)

        # Show scores
        self.logger.info("Model scores:")
        for model, score in model_scores.items():
            self.logger.info(f"{model}: {score}")

        best_model_name = max(model_scores, key=model_scores.get)
        best_model_index = [name for name, model in self.models].index(best_model_name)
        best_model = self.models[best_model_index][1]
        self.logger.info(f"Best Model: {best_model_name} with Score: {model_scores[best_model_name]}")

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        # y_pred = best_model.predict_proba(X_test) #  Probabilities

        # Evaluate model performance on the test set
        self.logger.info("Accuracy Score: %s", accuracy_score(y_test, y_pred))
        self.logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run()