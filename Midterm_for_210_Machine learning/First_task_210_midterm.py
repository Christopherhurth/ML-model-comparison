import time
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_NAME = "Diabetes Missing Data.csv"
TARGET = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_data(__file__, method):
    cwd = Path(__file__).parent.resolve()
    csv_path = cwd / DATA_NAME
    
    if not csv_path.exists():
        raise SystemExit(f"Cant find {csv_path}. Please check the file path.")
    
    data = pd.read_csv(csv_path)
    # print(data.shape)
    # print(data.columns.tolist())
    # print(data.isna().sum())
    
    target_column = data.columns[-1]
    data = data.dropna(subset=[target_column])
    
    if method == "drop":
        data = data.dropna()
    elif method == "zeros":
        data = data.fillna(0)
    elif method == "median":
        data = data.fillna(data.median(numeric_only=True))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'drop', 'zeros', or 'median'.")
    
    if len(data) == 0:
        raise SystemExit(f"No data left after applying method '{method}'. Please check the dataset.")
   
    data.to_csv(cwd / f"diabetes_{method}.csv", index=False)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    text_columns = X.select_dtypes(exclude="number").columns
    if len(text_columns) > 0:
        raise ValueError(f"these columns are text and need to be converted first: {list(text_columns)}")
    
    return X, y


def write_results(f, method, name, y_test, predictions, seconds):
    f.write(f"*** {method} | {name} ***\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    f.write(f"confusion matrix (rows = actual, columns = predicted):\n")
    f.write(f"{confusion_matrix(y_test, predictions)}\n")
    f.write(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Time: {seconds:.3f} seconds.\n\n")
    
    
def main():
    cwd = Path(__file__).parent.resolve()
    
    with open(cwd / "results.txt", "a") as f:
        for method in ["drop", "zeros", "median"]:
            X, y = get_data(__file__, method)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
            
            models = {
                "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(random_state=RANDOM_STATE)),
                "RandomForest":RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                "SVC": make_pipeline(StandardScaler(),SVC()),
            }
            
            for name, model in models.items():
                start = time.perf_counter()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                seconds = time.perf_counter() - start
                
                write_results(f, method, name, y_test, predictions, seconds)
                print("done:", method, name)
                
                
                
if __name__ == "__main__":
    main()
    
            