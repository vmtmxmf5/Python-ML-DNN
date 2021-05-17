# EDA 기본
train.head()
print(train.shape)
train.info()
train.describe()

## 장르별로 group by + 관객수 평균
train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num')

## 상관관계
 sns.heatmap(train.corr(), annot =True)
 

#############################################################

# 데이터전처리

## 결측값 확인
train.isna().sum()
train.isna().sum() / 행_개수

## 특정 열의 결측값 확인
train[train['dir_prev_bfnum'].isna()]
train[train['dir_prev_bfnum'].isna()]['dir_prv_num']
train[train['dir_prev_bfnum'].isna()]['dir_prv_num'].sum()

## 결측값 채워넣기
train['dir_prev_bfnum'].fillna(0, inplace = True)
test['dir_prev_bfnum'].fillna(0, inplace = True)

##############################################################

# 데이터모델링 1

## light GBM 모델링 (베이스 모델)
import lightgbm as lgb
model = lgb.LGBMRegressor(random_state = 777, n_estimators=1000)
## n_estimators : 모델 개수. 1000번 sequential하게 가중치를 둬서 모델을 만들겠다
## 잘 못맞힌 부분에 대하여 모델에서 가중치를 두어서 다시 학습을 진행시킨다
## 순차적으로(sequential) 모델을 계속해서 학습시킨다 (병렬적인 랜포랑은 다름) 

features = ['time', 'dir_prev_num', 'num_staff', 'num_actor']
target = ['box_off_num']

X_train, X_test, y_train = train[features], test[features], train[target]

model.fit(X_train, y_train)

from sklearn.model_selection import KFold
singleLGBM = submission.copy()
singleLGBM.head()
## 모델의 예측값(관객수)이 음수 = 예측이 잘못되었다

singleLGBM['box_off_num'] = model.predict(X_test)
singleLGBM.to_csv('singleLGBM.csv', index = False)


#############################################################

# 데이터모델링 2

## 교차검증 light GBM
## k=5등분 한 뒤, 1~4 학습 + 5 테스트, (1,2,3,5) 학습 + 4 테스트 ...... 이런 식으로 1 테스트 + 2~5 학습을 반복한다.
## 그 뒤 각각의 Acc의 평균을 Accuracy로 삼는다(앙상블 효과)

from sklearn.model_selection import KFold

k_fold = KFold(n_splits=5, shuffle=True, random_state=777)
## 데이터 생성순서에 따른 영향 제거를 위해 shuffle

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)

models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    models.append(model.fit(x_t, y_t, eval_set=(x_val, y_val), early_stopping_rounds=100, verbose=100))
    ## x_t, y_t 학습데이터셋
    ## eval_set validation셋
    ## early_stopping_rounds이란? 과적합을 방지하기 위해 100번의 학습동안 오차율 감소가 크지 않으면 학습 중단 시킴
    ## verbose란? 100번째마다 중간에 값 출력해줘

## 교차검증한 값을 pred에 담는다
preds = []
for model in models:
    preds.append(model.predict(X_test))
len(preds) 

kfoldLightGBM = submission.copy()

import numpy as np

## 교차검증 Acc 평균
KfoldLightGBM['box_off_num'] = np.mean(preds, axis = 0)

KfoldLightGBM.to_csv('kfoldLightGBM.csv', index=False)


#############################################################

# 데이터모델링 3

## feature engineering

from sklearn import preprocessing

## 문자 값을 숫자값으로 변경(피쳐 엔지니어링)
le = preprocessing.LabelEncoder()
train['genre'] = le.fit_transform(train['genre'])

train['genre']

test['genre'] = le.transform(test['genre'])

## genre를 바꿨으므로 다시 feature 설정
features = ['time', 'dir_prev_num', 'num_staff', 'num_actor', 'dir_prev_bfnum', 'genre']
X_train, X_test, y_train = train[features], test[features], train[target]

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)
models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    models.append(model.fit(x_t, y_t, eval_set=(x_val, y_val), early_stopping_rounds=100, verbose=100))

preds = []
for model in models:
    preds.append(model.predict(X_test))
len(preds)

feLightGBM = submission.copy()
feLightGBM['box_off_num'] = np.mean(preds, axis=0)
feLightGBM.to_csv('feLightGBM.csv', index=False)


#############################################################

# 데이터모델링 4

## grid search (모델 튜닝)

from sklearn.model_selection import GridSearchCV

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)

params = {
    'learning_rate': [0.1, 0.01, 0.003],
    'min_child_sample': [20, 30]
}

## scoring 오차율과 비슷
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  scoring='neg_mean_squared_error',
                  cv=k_fold)

gs.fit(X_train, y_train)

gs.best_params_

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000, learning_rate=0.003, min_child_samples=30)
