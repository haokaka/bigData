import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

test_data = pd.read_csv('G:\BigDataEvent\血糖预测\\f_test_a_20180204.csv',encoding='gb2312')
train_data = pd.read_csv('G:\BigDataEvent\血糖预测\\f_train_20180204.csv',encoding='gb2312')

def make_trainData(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()

    data = pd.concat([train, test])
    data = data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

train_feat, test_feat = make_trainData(train_data, test_data)
predictors = [f for f in test_feat.columns if f not in ['label', 'id']]

print('10 折训练开始：')

def my_loss(ground_truth, predictions):
    score = f1_score(ground_truth, predictions)
    return score

loss_f = make_scorer(my_loss, greater_is_better=True)

param_test1 = {'n_estimators': list(range(20,81,10))}
param_test2 = {'max_depth': list(range(4, 14, 2)), 'min_samples_split': list(range(20, 100, 10))}
param_test3 = {'min_samples_leaf': list(range(20, 81, 10))}
param_test4 = {'max_features': list(range(6, 20, 2))}
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.01,
                                                            n_estimators=600,
                                                            min_samples_split=70,
                                                            max_depth=6,
                                                            min_samples_leaf=20,
                                                            max_features=18,
                                                            random_state=10,
                                                            # subsample=0.8
                                                            ),
                       param_grid=param_test5,
                       scoring=loss_f,
                       cv=5)
gsearch1.fit(train_feat[predictors], train_feat['label'])
print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
