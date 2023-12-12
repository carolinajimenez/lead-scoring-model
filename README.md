# Lead Scoring Model

This project aims to develop a lead scoring model to predict the probability of conversion for potential clients into paying customers. The model is trained using data provided by the client from their Event Management SaaS application.

## Directory Structure
.       
├── README.md                           # Project documentation.        
├── data                                # Contains raw and processed data files.        
│   ├── interim             
│   │   ├── full_dataset.csv        
│   │   └── leads_data_cleaned.csv              
│   ├── processed               
│   │   └── full_dataset.csv                
│   └── raw             
│       ├── leads.csv               
│       └── offers.csv              
├── docs                
├── models              
├── notebooks               
│   └── data_preprocessing.ipynb                
├── reports             
│   ├── leads_report.html               
│   ├── model_training.log      
│   └── offers_report.html              
├── requirements.txt                    # Required dependencies for the project.        
└── src                                 # Contains source code for the project.        
    ├── app.py              
    ├── models              
    │   ├── __init__.py             
    │   ├── predict_model.py                
    │   └── train_model.py              
    └── utils               
        └── logger.py               


## Data Description

### leads.csv
- **Id:** Unique identifier for the lead.
- **First Name:** Lead's first name.
- **Use Case:** Type of use case for the potential client.
- **Source:** Lead source (e.g., Inbound, Outbound).
- **Status:** Current status of the lead.
- **Discarded/Nurturing Reason:** Reason for lead discard or nurturing.
- **Acquisition Campaign:** Acquisition campaign that generated the lead.
- **Created Date:** Lead creation date.
- **Converted:** Target variable, indicating whether the lead converted (1) or not (0).
- **City:** City of the lead.

### offers.csv
- **Id:** Unique identifier for the offer.
- **Use Case:** Type of use case for the offer.
- **Status:** Current status of the offer.
- **Created Date:** Offer creation date.
- **Close Date:** Offer closing date.
- **Price:** Offer price.
- **Discount code:** Applied discount code.
- **Pain:** Customer potential's pain level.
- **Loss Reason:** Reason for offer loss.

## Data Preparation and Modeling

1. **Data Fusion:** The `leads.csv` and `offers.csv` datasets were merged using the unique identifier (`Id`).

2. **Handling Missing Data:**
   - Removed rows from `leads.csv` where there were null values in the `Id` column.
   - Drop `First Name`, `Use Case`, `Created Date`, `Status`, `Converted` columns from `leads.csv`.
   - Drop `Id`, `Discarded/Nurturing Reason`, `Acquisition Campaign` columns from merged-data.
   - Imputed missing values in `Loss Reason` column according `Status` values from merged-data.
   - Imputed missing values in categorical columns with the mode from merged-data.
   - Imputed missing values in numerical columns with the mean from merged-data.

3. **Mapping target column values:**
   - Group the minority classes into a new class called Other.

4. **Label Encoding:**
   - Applied Label Encoding to categorical columns: `Source`, `City`, `Loss Reason`, `Pain`, `Discount code`, `Status`, `Use Case`.

4. **Data Scaling:**
   - Used StandardScaler to scale numerical features in the dataset.

5. **Model Selection:**
   - Explored various classification algorithms, including Random Forest, Adaboost, Extra Trees, Bagging Classifier, Gradient Boosting, Decision Tree, Naive Bayes, KNN, Logistic Regression, SGD Classifier, MLP Classifier, and SVM.

6. **Model Evaluation:**
   - Compared model performance using cross-validation scores.

## Dependencies

Ensure you have the necessary dependencies installed by running:

```bash
pip install -r requirements.txt
```

## Usage

* To pre-process data, run the `data_preprocessing.py` script.

```bash
python3 src.eda.data_preprocessing.py
```

* To train and evaluate the models, run the `train_model.py` script. This will load the processed dataset, split it into training and testing sets, train multiple models, and display their cross-validation scores.

```bash
python3 -m src.models.train_model
```

* To run application with Shimoku API

```bash
python3 src/app.py
```

## Results

- **Data Transformation:** Classes in the "Status" column were grouped into three categories - "Closed Won," "Closed Lost," and "Other" to address class imbalance.

- **Best Model:** The GradientBoosting model achieved the highest cross-validation score (0.91) and was selected for further evaluation.

- **Classification Results:**
  - **Accuracy:** 90.4%
  - **Precision, Recall, F1-score:** Detailed classification metrics for each class are provided in the updated classification report.
