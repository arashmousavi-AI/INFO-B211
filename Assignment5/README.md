# Assignment 5 

## Purpose of the Program
The purpose of this project is to apply machine learning classification techniques leveraging the Scikit-Learn library in Python. 
The program uses the built-in Breast Cancer dataset to train and evaluate several classification models that predict whether a tumor is malignant or benign.

Three machine learning models were implemented:
- Logistic Regression
- K Nearest Neighbors
- Decision Tree Classifier

Each model was trained using a training dataset and evaluated using a testing dataset. Different parameter values were tested for each model to improve performance
and identify the best-performing configuration

## Input:
The program uses the Breast Cancer Wisconsin dataset provided by Scikit-Learn.
The dataset contains:
- 569 samples
- 30 numerical features

These features describe characteristics of cell nuclei present in digitized breast cancer images.

Examples of input features include:
- mean radius
- mean texture
- mean permeter
- mean area
- worst radius
- worst texture
- worst concave points

The dataset also includes a target variable that indicates whether the tumor is:
- Malignant
- Benign
The dataset is automatically loaded using the Scikit-Learn library.

## Expected Output:
The program trains three different classification models and evaluates their performance.

The output includes:
- Model accuracy
- Precision score
- Recall score
- F1 score
- Confusion matrix

Each model is evaluated using these metrics to determine how accurately it predicts tumor classifications.

The program also prints a summary table comparing all model configurations and identifies the best performing model overall.

In addition, feature importance values from the Decision Tree model are displayed to show which dataset features have the strongest influence on predictions.

## Model Comparison and Results
Three machine learning classification models were tested: Logistic Regression, K-Nearest Neighbors, and Decision Tree. For each model, different parameter values were tested to determine the best performing configuration. Performance was evaluated using accuracy, precision, recall, and F1 score. The best overall model was Logistic Regression with parameters max_iter = 2000 and C = 0.1. This model achieved an accuracy of 0.9649 and the highest F1 score of 0.9722. Although KNN and Decision Tree also performed well, Logistic Regression provided the best balance between precision and recall. Feature importance analysis from the Decision Tree model showed that mean concave points and worst texture were among the most important features for predicting whether a tumor is malignant or benign.

## Type of Execution
The program involves the following types of execution:

Sequential Execution:
The code runs sequentially from loading the dataset to training the models and evaluating their results.

Reusable Execution:
A reusable evaluation function is used to calculate accuracy, precision, recall, and F1 score for each model.

Repeated Execution:
Loops are used to test different parameter configurations for each machine learning model.

Conditional Execution:
Conditional statements are used to determine which model configuration produces the best performance based on the F1 score.

## The Possible Improvements: 
The program could be improved in several ways:

- Applying feature scaling using StandardScaler to improve the performance of models like KNN and Logistic Regression.
- Using cross-validation to obtain more reliable performance estimates.
- Implementing GridSearchCV to automatically tune hyperparameters.
- Testing additional models such as Random Forest, Support Vector Machines, or Gradient Boosting.
- Visualizing results using plots such as ROC curves or feature importance charts.

These improvements could help further optimize model performance and provide deeper insight into the dataset.
