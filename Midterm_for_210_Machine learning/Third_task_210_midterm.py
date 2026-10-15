import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

DATA_NAME = "college_student_placement_dataset.csv"
RESULTS_NAME = "results_for_student_placement.txt"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_data(__file__):
    cwd = Path(__file__).parent.resolve()
    csv_path = cwd / DATA_NAME

    if not csv_path.exists():
        raise SystemExit(f"Can't find {csv_path}. Put the csv in the same folder as this script.")

    data = pd.read_csv(csv_path)
    data = data.dropna()
    data = data.drop(columns=["College_ID"])

    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = (data[target_column] == "Yes").astype(int)

    X = pd.get_dummies(X, dtype=int)
    return X, y


def write_results(f, name, y_test, predictions, seconds):
    f.write(f"*** {name} ***\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    f.write("Confusion matrix (rows = actual, columns = predicted):\n")
    f.write(f"{confusion_matrix(y_test, predictions)}\n")
    f.write(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Time: {seconds:.3f} seconds\n\n")


def write_extras(f, forest, X, y):
    importances = pd.Series(forest.feature_importances_, index=X.columns)
    f.write("Random forest feature importances:\n")
    f.write(f"{importances.sort_values(ascending=False).round(4)}\n\n")

    scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        X, y, cv=5, scoring="f1")
    f.write(f"Random forest 5-fold cross-validation F1: {scores.round(4)}\n")
    f.write(f"Mean: {scores.mean():.4f}\n\n")


def main():
    cwd = Path(__file__).parent.resolve()
    X, y = get_data(__file__)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    models = {
        "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(random_state=RANDOM_STATE)),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVC": make_pipeline(StandardScaler(), SVC()),
    }

    with open(cwd / RESULTS_NAME, "a") as f:
        f.write(f"Share of students placed: {y.mean():.3f}\n")
        f.write(f"Always guessing 'No' would score {1 - y.mean():.3f} accuracy\n\n")

        for name, model in models.items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            seconds = time.perf_counter() - start

            write_results(f, name, y_test, predictions, seconds)
            print("done:", name)

        write_extras(f, models["RandomForest"], X, y)


if __name__ == "__main__":
    main()
