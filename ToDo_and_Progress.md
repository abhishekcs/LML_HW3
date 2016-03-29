To Do:

* Train and ensemble all combinations of these models
    - Logistic, XGB
    - With One-hot, without One-hot
    - With/without role-1 and role2 (suggested here : https://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5283/winning-solution-code-and-methodology)

Progress:

* Logistic without one-hot : 51.924
* XGB without one-hot : 87.25
* XGB with one-hot : 88.48 (code in hw3p2-XGB-OneHot.ipynb, predictions in XGB-One-Hot.csv)
* L1-Logistic with one-hot & new features : 89.613 (code in hw3p1.py, predictions in hw3p1.csv)
* 90.43 private score with the following ensemble : ['./XGB-One-Hot.csv', './hw3p1_arun_896.csv', './hw3rf.csv', './hw3p2_arun_883.csv']
