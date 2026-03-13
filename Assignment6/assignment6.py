import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


#loading dataset
diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#the evaluation function
def evaluate_model(model_name, parameters, y_true, y_pred):

    ev = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("model:", model_name)
    print("parameters:", parameters)
    print("explained Variance:", ev)
    print("mean Squared Error:", mse)
    print("R2 Score:", r2)

    return {
        "model": model_name,
        "parameters": str(parameters),
        "explained Variance": ev,
        "mean Squared Error": mse,
        "R2 Score": r2
    }

results = []


#linear Regression experiments
lr_configs = [
    {"fit_intercept": True},
    {"fit_intercept": False}
]

for config in lr_configs:

    model = LinearRegression(
        fit_intercept=config["fit_intercept"]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append(
        evaluate_model("Linear Regression", config, y_test, pred)
    )


#random Forest experiments
rf_configs = [
    {"n_estimators": 50, "random_state": 42},
    {"n_estimators": 75, "random_state": 42},
    {"n_estimators": 100, "random_state": 42}
]

for config in rf_configs:

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append(
        evaluate_model("Random Forest Regressor", config, y_test, pred)
    )


#SVR
svr_configs = [
    {"kernel": "rbf", "C": 1.0},
    {"kernel": "rbf", "C": 10.0},
    {"kernel": "linear", "C": 1.0}
]

for config in svr_configs:

    model = SVR(
        kernel=config["kernel"],
        C=config["C"]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append(
        evaluate_model("Support Vector Regressor", config, y_test, pred)
    )


#create summary table
results_df = pd.DataFrame(results)

print("the summary table")
print(results_df)


#finding the best model
best_model = results_df.loc[results_df["R2 Score"].idxmax()]

print("The best model")
print(best_model)