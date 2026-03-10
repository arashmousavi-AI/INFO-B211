import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#loading the breast cancer dataset

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("number of samples:", X.shape[0])
print("number of features:", X.shape[1])
print("target names:", cancer.target_names)
print("feature names:")
print(cancer.feature_names)


#spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\ntraining set shape:", X_train.shape)
print("testing set shape:", X_test.shape)


#evaluating a model

def evaluate_model(model_name, parameters, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("model:", model_name)
    print("parameters:", parameters)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 Score:", f1)
    print("confusion Matrix:")
    print(cm)

    return {
        "model": model_name,
        "parameters": str(parameters),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1 Score": f1
    }

#logistic Regression with different settings
results = []

log_configs = [
    {"max_iter": 1000, "C": 1.0, "solver": "liblinear"},
    {"max_iter": 2000, "C": 0.1, "solver": "liblinear"},
    {"max_iter": 3000, "C": 10.0, "solver": "liblinear"}
]

for config in log_configs:
    log_model = LogisticRegression(
        max_iter=config["max_iter"],
        C=config["C"],
        solver=config["solver"],
        random_state=42
    )

    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)

    result = evaluate_model("Logistic Regression", config, y_test, log_pred)
    results.append(result)


#KNN with different settings 
knn_configs = [
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 7, "weights": "distance"}
]

for config in knn_configs:
    knn_model = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        weights=config["weights"]
    )

    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)

    result = evaluate_model("K-Nearest Neighbors", config, y_test, knn_pred)
    results.append(result)


#decision Tree with different settings
dt_configs = [
    {"max_depth": 3, "criterion": "gini"},
    {"max_depth": 5, "criterion": "gini"},
    {"max_depth": 5, "criterion": "entropy"}
]

best_dt_model = None
best_dt_f1 = -1

for config in dt_configs:
    dt_model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        criterion=config["criterion"],
        random_state=42
    )

    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    result = evaluate_model("Decision Tree", config, y_test, dt_pred)
    results.append(result)

    if result["f1 Score"] > best_dt_f1:
        best_dt_f1 = result["f1 Score"]
        best_dt_model = dt_model


# the summary table of all the model results
results_df = pd.DataFrame(results)

print("SUMMARY TABLE")
print(results_df)


#best model overall
best_model_row = results_df.loc[results_df["f1 Score"].idxmax()]

print("BEST MODEL OVERALL")
print(best_model_row)


#feature importances from best Decision Tree
if best_dt_model is not None:
    feature_importance_df = pd.DataFrame({
        "feature": cancer.feature_names,
        "importance": best_dt_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("BEST DECISION TREE FEATURE IMPORTANCES")    
    print(feature_importance_df.head(10))