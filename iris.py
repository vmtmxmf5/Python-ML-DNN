# Iris

import sklearn 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


## 데이터전처리
iris = load_iris()    #데이터 객체 불러오기
iris_data = iris.data  

iris_label = iris.target
print('target 값', iris_label)
print('iris target이름', iris.target_names)

iris_df = pd.DataFrame(data=iris_data,
                       columns=iris.feature_names)
iris_df['label'] = iris.target #y열 추가
iris_df.head(3)


### array를 넣어야 함
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                 test_size = 0.2, random_state=11)


## Decision Tree Classifier 
dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

## Acc 계산

from sklearn.metrics import accuracy_score

print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')


## K-fold 교차검증
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5)
cv_accuracy = []

n_iter = 0

features = iris.data
label = iris.target

for train_index, test_index in kfold.split(features):
    X_trian, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'횟수 : {n_iter}, 정확도 : {accuracy}, 학습 크기 : {train_size}')    
    print(f'횟수 : {n_iter}, 검증 : {test_index}')
    cv_accuracy.append(accuracy)



## Stratified K-fold 검증
### y값 중 한 그룹의 count가 부족하면 SKF!!!
### group별로 쪼개서 각각 k-fold cv를 진행

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_ix, test_ix in skf.split(features, label):
    X_train, X_test = features[train_ix], features[test_ix]
    y_train, y_test = label[train_ix], label[test_ix]
    
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    cv_acc.append(accuracy)
    
    
    
    
    

    
    