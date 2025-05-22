# Kaggle Playground Series S4E8 - Binary Prediction of Poisonous Mushrooms

This repository contains the code and analysis for the Kaggle Playground Series Season 4 Episode 8 competition: [Binary Prediction of Poisonous Mushrooms](https://www.kaggle.com/competitions/playground-series-s4e8/overview). The goal of this competition is to build a model that can accurately predict whether a mushroom is **poisonous (p)** or **edible (e)** based on a variety of features like cap shape, color, gill characteristics, and stem properties.

---

## Notebook Content

The [predicting_poisonous_mushrooms.ipynb](predicting_poisonous_mushrooms.ipynb) notebook walks through the following steps:

1. **Data Loading and Exploration:**
    - Loading the `train.csv` and `test.csv` datasets using pandas.
    - Displaying the shape of the datasets and the first few rows of the training data.

2. **Missing Value Handling:**
    - Identifying columns with missing values and their counts.
    - Implementing a strategy to handle missing data:
        - Dropping rows with missing values in columns with a small number of missing entries.
        - Filling missing values with the string "missing" for columns with a large number of missing entries.

3. **Value Counts Analysis:**
    - Defining a function to calculate and display value counts and percentages for each column to understand data distribution.
    - Applying this function to the training dataset to analyze the distribution of features.

4. **Filtering Rare Categories:**
    - Identifying and removing rows containing rare categories (occurring less than 200 times) in each feature column (excluding 'id' and 'class') to reduce noise and improve model generalization.

5. **Feature Engineering and Encoding:**
    - Separating the target variable ('class') from the features in the training dataset.
    - Performing one-hot encoding on categorical features to convert them into a numerical format suitable for machine learning models.

6. **Preprocessing the Test Data:**
    - Applying similar preprocessing steps (missing value handling and one-hot encoding) to the `test.csv` dataset to ensure consistency with the training data.
    - Handling potential discrepancies in categorical features between the training and test sets by aligning columns after one-hot encoding.

7. **Correlation Analysis and Feature Selection:**
    - Calculating the correlation matrix for the numerical features.
    - Generating a **heatmap** of correlations to visually identify relationships between features.
    - Saving the heatmap as an image (`mapa_correlacao.png`) for reference.
    - Identifying **highly correlated feature pairs** (with Pearson correlation `r > 0.85`) to flag for potential removal or collinearity analysis.
    - Example: `cap-diameter` and `stem-width` were found to be highly correlated (`r = 0.83`), suggesting caution during feature selection for linear models.

    ![Mapa de Correlação](mapa_correlacao.png)

8. **Data Splitting into Training and Validation Sets:**
    - Splitting the preprocessed training data into training and validation sets using `train_test_split` from `sklearn.model_selection`.
    - Using stratified splitting to maintain the class distribution in both sets.

9. **Feature Scaling:**
    - Applying `StandardScaler` from `sklearn.preprocessing` to standardize numerical features in both the training and validation sets to improve model performance and convergence.

10. **Model Training and Evaluation:**
    - Defining functions `train_model` and `evaluate_model` to streamline the process of training and evaluating different classifiers.
    - Training and evaluating three different machine learning models:
        - **Logistic Regression:** A linear model used as a baseline.
        - **Random Forest Classifier:** An ensemble method known for its robustness and performance on tabular data.
        - **XGBoost Classifier:** A gradient boosting algorithm, often achieving state-of-the-art results in classification tasks.
    - Evaluating model performance using metrics such as Accuracy, F1-Score, Confusion Matrix, and AUC-ROC curve.

11. **Prediction and Submission File Generation:**
    - Scaling the test dataset using the `StandardScaler` fitted on the training data.
    - Using the trained XGBoost model (the best performing model) to predict classes for the test dataset.
    - Decoding the numerical predictions back to the original class labels ('e' or 'p').
    - Creating a submission-ready CSV file (`predictions_xgb.csv`) with 'id' and 'class' columns.

---

## Models and Performance

The following models were trained and evaluated:

| Model                    | Accuracy | F1 Score | AUC-ROC Score |
|--------------------------|----------|----------|----------------|
| Logistic Regression      | 0.8897   | 0.8897   | 0.9373         |
| Random Forest Classifier | 0.9920   | 0.9920   | 0.9966         |
| XGBoost Classifier       | 0.9924   | 0.9924   | 0.9968         |

The **XGBoost Classifier** achieved the best performance on the validation set with an Accuracy of 0.9924 and an AUC-ROC Score of 0.9968. This model was used to generate the final predictions for the test set.

---

## Additional Notes

- The correlation analysis and variable filtering were useful to improve understanding of the dataset structure and to guide model design decisions.
- The generated heatmap (`mapa_correlacao.png`) can be helpful for further feature selection, especially if dimensionality reduction or regularization is needed.
