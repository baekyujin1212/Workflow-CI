
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================
# LOAD DATA
# ==========================

X_train = pd.read_csv(
    "bank marketing_preprocessing/X_train.csv"
)

X_test = pd.read_csv(
    "bank marketing_preprocessing/X_test.csv"
)

y_train = pd.read_csv(
    "bank marketing_preprocessing/y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    "bank marketing_preprocessing/y_test.csv"
).values.ravel()


# ==========================
# CEK UKURAN DATA
# ==========================

print("Ukuran dataset:")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ==========================
# MLFLOW AUTOLOG
# ==========================

mlflow.set_experiment(
    "RandomForest Classification"
)

mlflow.sklearn.autolog()


# ==========================
# TRAIN MODEL
# ==========================

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    print("Training model...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("Accuracy:", accuracy)

print("Training selesai")

