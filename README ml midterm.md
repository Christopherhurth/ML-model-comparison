# Machine Learning Midterm: Model Comparison

Midterm project for AICC-210 (Machine Learning), College of Western Idaho.

Three classification tasks comparing traditional machine learning models with a neural network, with a focus on judging models fairly, not just by accuracy.

## Task 1: Diabetes data with missing values (`First_task_210_midterm.py`)

The diabetes dataset has missing values. I built three versions of it to see how the cleanup choice affects results:

- **drop:** remove rows with missing values (`diabetes_drop.csv`)
- **zeros:** fill missing values with 0 (`diabetes_zeros.csv`)
- **median:** fill missing values with each column's median (`diabetes_median.csv`)

I then trained an SGD classifier, a Random Forest and an SVC on each version (80/20 train/test split, scaled features) and recorded accuracy, precision, recall, the confusion matrix and training time. Results: `results.txt`.

## Task 2: PyTorch neural network (`Second_task_210_midterm.py`)

I trained a small PyTorch neural network (one hidden layer of 64 units, 200 epochs) on the same three datasets and compared it with Task 1. Results: `results_pytorch.txt`.

**Finding:** the neural network performed about the same as the other models, not clearly better or worse. It did slightly better on the median data, worse than Random Forest on the dropped data, and tied on the zeros data. Most gaps were only a few patients, so they're likely noise. On a small, untuned dataset the network had no real edge and was slower than most of the other models.

## Task 3: Student placement (`Third_task_210_midterm.py`)

I predicted whether a college student gets placed in a job using the same three models. I dropped `College_ID` (a label with no predictive value) and converted the Yes/No columns to 0/1.

Only about **17%** of students were placed, so a model that always guessed "No" would already be 83% accurate. That's why I judged the models on **precision and recall**, not only accuracy.

- **Random Forest** performed best (about 99.7% recall), confirmed with 5-fold cross-validation.
- **SGD classifier** missed about 44% of placed students.
- The most important features were communication skills, IQ, projects completed and CGPA. Placement seems to depend on cut-offs, which is what decision trees are good at.

Scores this high usually mean a very clean or synthetic dataset, so I'd expect lower results on real-world data.

Results: `results_for_student_placement.txt`. Write-ups: `Midterm_write_up.md` and `Midterm_Explanation_Prblm_3_WriteUp.md`.

## Tech stack

Python, pandas, NumPy, scikit-learn, PyTorch

## Running it

```bash
pip install pandas numpy scikit-learn torch
python First_task_210_midterm.py
python Second_task_210_midterm.py
python Third_task_210_midterm.py
```

The scripts expect the original datasets (`Diabetes Missing Data.csv` and `college_student_placement_dataset.csv`) in the same folder. They aren't included here; the cleaned diabetes files produced by Task 1 are.

## Note on AI assistance
Claude was used to help me work through the project. I ran and checked the results myself.

