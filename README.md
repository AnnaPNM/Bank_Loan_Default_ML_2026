# Bank_Loan_Default_ML_2026
Current repository presents the result of team "Financial predictors" work in Skoltech ML course 2026


- Anna Ponomareva (Life Sciences)
- Uliana Orlova (Life Sciences)
- Ahmed Eltwam (Engineering Systems)
- Kamal Hammad (Engineering Systems)
- Dmitry Davydov (Petroleum Engineering)


The current work aims to find a model with the best performance for the prediction of bank loan default based on provided bank real data (Home Credit Default Risk. (2018). Kaggle. https://www.kaggle.com/competitions/home-credit-default-risk/data). We have applied diverse ML methods, such as: Logistic Regression, SVM, Random Forest (with several boosting techniques), MLP; and compared their metrics. The current work demonstrates, that class imbalance is a critical problem in machine learning, moreover, usually in real data classes are not balanced. We have tried to solve this problem by several methods, such as weighted samples, SMOTE and SMOTEENN. The best ROC-AUC scores (0.77-0.78) were obtained on Decision Trees with different boosting techniques. MLP has demonstrated the highest Recall (0.91) and F1-score (0.66) metrics.


Link for the EDA preprocessed training and test data files: 

https://drive.google.com/drive/folders/154rtT1QPJjiS-4HCxboi_grAZ_9hjYud?usp=sharing 
