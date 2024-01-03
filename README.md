# Cancer-Diagnosis-Prediction

## Problem Statement

This repository contains code for classifying breast cancer using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). The dataset includes various features computed from breast mass images, and the task is to predict whether a given tumor is benign or malignant.

## Data Exploration

The dataset is loaded using the `load_breast_cancer` function from the `sklearn.datasets` module, and then converted into a pandas DataFrame. Here are some initial steps performed on the dataset:

- Displaying the shape of the dataset and the number of unique values in each column.
- Grouping the data by the target variable and calculating mean values for the numerical columns.

## Data Visualization

Data visualization is an important step to gain insights and understand the patterns in the data. Here are some visualizations done on the breast cancer dataset:

- Pair plot: A scatter plot matrix showing the relationship between pairs of features, with different colors representing the target variable.
- Count plot: A bar plot showing the distribution of the target variable.
- Histograms: Plots showing the distribution of numerical features using histograms.
- Box plots: Plots showing the distribution of numerical features using box plots.
- Scatter plot: A scatter plot showing the relationship between 'mean area' and 'mean smoothness', with different colors representing the target variable.
- Correlation heatmap: A heatmap showing the correlation between numerical features.

## Handling High Correlations

High correlation between features can lead to multicollinearity and affect the performance of machine learning models. The `high_correlated_cols` function calculates the correlation matrix and drops features with a correlation threshold higher than 0.90. It also provides an option to plot the correlation matrix as a heatmap.

## Handling Outliers

Outliers can have a significant impact on the performance of machine learning models. To identify and handle outliers, the following steps are performed:

- Calculating the lower and upper limits for outlier detection using the interquartile range.
- Checking for outliers in each numerical column and printing the results.
- Replacing outliers with the corresponding limits.

## Preprocessing and Model Training

The dataset is split into training and test sets using the `train_test_split` function from `sklearn.model_selection`. A support vector machine (SVM) model is trained on the training set and evaluated on the test set using classification metrics such as confusion matrix and classification report.

## Hyperparameter Tuning

Hyperparameter tuning is performed using the `GridSearchCV` function from `sklearn.model_selection`. The SVM model is tuned by searching over a grid of hyperparameters to find the best combination. The best parameters and estimator are printed, and the model is evaluated on the test set using classification metrics.

Feel free to explore the code and adapt it to your own projects! in the project is for demonstration purposes. In practice, the dataset and parameters may vary, and additional preprocessing steps and evaluations may be necessary.