from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATASET_FILES = {
    "A_dropped_rows": "diabetes_drop.csv",
    "B_zeros": "diabetes_zeros.csv",
    "C_median": "diabetes_median.csv",
}


RESULTS_NAME = "results_pytorch.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.20
HIDDEN = 64
EPOCHS = 200
LEARNING_RATE = 0.001




def get_data(__file__, filename):
    cwd = Path(__file__).parent.resolve()
    csv_path = cwd / filename
    
    if not csv_path.exists():
        raise SystemExit(f"Cant find {csv_path}. Please check the file path.")
    
    data = pd.read_csv(csv_path)
    
    target_column = data.columns[-1]
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x):
         return self.net(x)




def write_results(f, filename, y_test, predictions, seconds):
    f.write(f"*** {filename} | Pytorch MLP ***\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    f.write(f"confusion matrix (rows = actual, columns = predicted):\n")
    f.write(f"{confusion_matrix(y_test, predictions, labels=[0, 1])}\n")
    f.write(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}\n")
    f.write(f"Time: {seconds:.3f} seconds.\n\n")






def main():
    cwd = Path(__file__).parent.resolve()
    files = ["diabetes_drop.csv", "diabetes_zeros.csv", "diabetes_median.csv"]
    
    with open(cwd / RESULTS_NAME, "a") as f:
        for filename in files:
            X,y = get_data(__file__, filename)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            X_train = torch.tensor(X_train, dtype=torch.float32)   
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.long)
            
            torch.manual_seed(RANDOM_STATE)
            model = MLP(X_train.shape[1], HIDDEN, 2)
            loss_funtion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            start = time.perf_counter()
            for epoch in range(EPOCHS):
                optimizer.zero_grad()
                scores = model(X_train)
                loss = loss_funtion(scores, y_train)
                loss.backward() 
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                predictions = model(X_test).argmax(dim=1).numpy()
            seconds = time.perf_counter() - start
            
            write_results(f, filename, y_test, predictions, seconds)
            print("Done:", filename)
                

if __name__ == "__main__":
    main()
