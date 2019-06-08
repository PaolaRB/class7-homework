# Loading data using sklearn dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pandas as pd

cancer = load_breast_cancer()
column_names = cancer.feature_names
X = cancer.data
y = cancer.target

new_column = np.array(['target'])
feature_names = cancer.feature_names

print(np.unique(y))
print(f'sklearn cancer dataset X shape: {X.shape}')
print(f'sklearn cancer dataset y shape: {y.shape}')
print(f'keys: {cancer.keys()}')
print(f'data: {cancer.data}')
print(f'target: {cancer.target}')
print(f'Describe: {cancer.DESCR}')
print(f'Filename: {cancer.filename}')
print(np.append(new_column, feature_names))


# ************************************
# ********** Split dataset ***********
# ************************************

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Printing splitted datasets
print(f'X_train.shape : {X_train.shape}, y_train.shape : {y_train.shape}')
print(f'X_test.shape : {X_test.shape}, y_test.shape : {y_test.shape}')

# ************************************
# ********** Training model***********
# ************************************

lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
lr.fit(X_train, y_train)

print(f'Intercept per class: {lr.intercept_}\n')
print(f'Coeficients per class: {lr.coef_}\n')
print(f'Available classes : {lr.classes_}\n')
print(f'Named Coeficients for class 0: {pd.DataFrame(lr.coef_[0], column_names)}\n')
print(f'Number of iterations generating model : {lr.n_iter_}')

# ************************************
# ********** Predicting the results **
# ************************************

predicted_values = lr.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred:{predicted} {"is different " if real != predicted else ""}')

# ************************************
# ********** Accuracy SCore **********
# ************************************
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# ************************************
# ******** Printing Reports **********
# ************************************
print('Classification report\n')
print(classification_report(y_test, predicted_values))
print('Confusion matrix\n')
print(confusion_matrix(y_test, predicted_values))
print('Overal f1-score\n')
print(f1_score(y_test, predicted_values, average="macro"))


# ************************************
# ******** Cross Validation **********
# ************************************
print(cross_val_score(lr, X, y, cv=10))
cv = ShuffleSplit(n_splits=5)
print(cross_val_score(lr, X, y, cv=10))
