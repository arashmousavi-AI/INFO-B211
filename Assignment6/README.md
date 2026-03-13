# Assignment 6 | Scikit-Learn Regression Assignment 

## Purpose of the Program
The purpose of this project is to implement and evaluate multiple machine learning regression models using the Scikit-Learn library in Python. 
The program uses the built-in Diabetes dataset to train and evaluate models that predict disease progression based on several medical measurements.

Three regression models were implemented and compared:
- Linear Regression
- Random Forest Regresssor
- Support Vector Regressor (SVR)

Different parameter configurations were tested for each model in order to explore how hyperparameter changes affect model performance.
The models were evaluated using several regression metrics to determine which model performed best.

## Input:
The program uses the built-in Diabetes dataset available in Scikit-Learn.
The dataset contains:
- 442 samples
- 10 numerical features

These features represent various medical measurements related to diabetes, including:
- age
- sex
- body mass index (BMI)
- blood pressure
- blood serum measurement

The target variable represents a quantitative measure of disease progression one year after baseline.

The dataset is automatically loaded using the Scikit-Learn library, so no external data files or manual user input are required.

## Expected Output:
The program trains three regression models and evaluates their performance using several regression metrics.

The output includes:
- Explained Variance Score
- Mean Squared Error (MSE)
- R² Score

Multiple parameter configurations are tested for each model. The results are printed for every configuration and summarized in a comparison table.
The program also identifies the best-performing model based on the evaluation metrics.

In this experiment, the Support Vector Regressor with an rbf kernel and C = 10.0 produced the best overall performance, achieving the highest explained 
variance score and R² score while also producing the lowest mean squared error.

## Model Comparison and Results
Three machine learning regression models were tested: Linear Regression, Random Forest Regressor, and Support Vector Regressor (SVR). For each model, 
different parameter values were tested to determine the best performing configuration. Performance was evaluated using explained variance score, mean squared error (MSE), 
and R² score. 
The best overall model was Support Vector Regressor (SVR) with parameters kernel = 'rbf' and C = 10.0. This configuration achieved the highest explained variance score of approximately 0.501, 
the lowest mean squared error of approximately 2679, and the highest R² score of approximately 0.494. 
Linear Regression also performed well, achieving an R² score of approximately 0.453 
with a mean squared error of around 2900. Random Forest Regressor produced similar results with an R² score of approximately 0.443 depending on the number of trees used.
The results indicate that the tuned Support Vector Regressor model captured the underlying patterns in the dataset more effectively than the other models, resulting in better predictive performance.
Testing multiple parameter configurations helped improve model performance and demonstrated how hyperparameter tuning can impact regression results.

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
- Applying feature scaling using StandardScaler to improve model performance, especially for SVR.
- Using cross-validation to obtain more reliable performance estimates.
- Implementing GridSearchCV to automatically search for optimal hyperparameters.
- Testing additional regression models such as K-Nearest Neighbors Regressor, Decision Tree Regressor, or Gradient Boosting Regressor.
- Adding visualizations such as prediction vs. actual value plots or residual error plots.

These improvements could further enhance the predictive performance of the models and provide deeper insight into the dataset.
