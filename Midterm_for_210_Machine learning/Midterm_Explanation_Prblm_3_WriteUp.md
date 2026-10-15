^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PROBLEM 3 Explanation write up
*********************************************************************************************************************************************

For Problem 3 I ran SGDClassifier, Random Forest and SVC on the same 80/20 split I used in Problem 1, and the results are in results_for_student_placement.txt. I went this way mostly because it was the quickest and easiest option, since I could reuse the structure from the earlier problems, and then I checked that it also made sense for this data.

I dropped College_ID because it's only a label and doesn't help predict placement, and I converted the Yes/No columns to 0/1. Since only about 17% of students are placed, I judged the models on precision and recall as well as accuracy, because always guessing "No" would already be 83% accurate.

Random Forest worked best (about 99.7% recall), while SGDClassifier missed about 44% of the placed students. I think Random Forest fit because placement seems to depend on cut-offs (IQ, CGPA, communication skills), and trees split on cut-offs. I would like to note as well that Claude did help me through this whole project.