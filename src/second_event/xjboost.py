import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import process_eigen

def convert(x):
    if x >= 0.5:
        return 1
    return 0

def convert1(x):
    res = []
    for each in x:
        if each >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

path = 'G:\BigDataEvent\血糖预测\\'
# test = pd.read_csv(path + 'f_test_a_20180204.csv',encoding='gb2312')
# train = pd.read_csv(path + 'f_train_20180204.csv', encoding='gb2312')
#
# train_data, test_data = process_eigen.process_eigen(train, test)
train_data = pd.read_csv(path + 'process_train_fb.csv', encoding='gb2312')
test_data = pd.read_csv(path + 'process_test_fb.csv', encoding='gb2312')

print('开始训练。。。')
predictors = [f for f in test_data.columns if f not in ['label', 'id']]
xgb_test = xgb.DMatrix(test_data[predictors])
params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':2,
    'lambda':10,
    'gamma':0.15,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'min_child_weight':3,
    'eta': 0.1,
    'seed':0,
    'nthread':8,
     'silent':1}


train_preds = np.zeros(train_data.shape[0])
test_preds = np.zeros((test_data.shape[0], 5))
kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=84)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_data.iloc[train_index]
    valid_feat = train_data.iloc[test_index]
    xgb_train = xgb.DMatrix(train_feat[predictors], label=train_feat['label'])
    xgb_valid = xgb.DMatrix(valid_feat[predictors])
    watchlist = [(xgb_train, 'train')]

    estimator = xgb.train(params,xgb_train,num_boost_round=250,evals=watchlist)

    train_preds[test_index] += estimator.predict(xgb_valid)
    test_preds[:,i] = estimator.predict(xgb_test)

print('线下得分：f1', f1_score(train_data['label'], convert1(train_preds)))
print('线下得分：precision', precision_score(train_data['label'], convert1(train_preds)))
print('线下得分：recall', recall_score(train_data['label'], convert1(train_preds)))
# submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
# submission = submission['pred'].apply(lambda x : convert(x))
# submission.to_csv(r'G:\machine learning\diabetes\classification\submission\xgb_0306_{}.csv'.format('2'), header=None,
#                   index=False)