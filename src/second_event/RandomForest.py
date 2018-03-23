import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import process_eigen
import process_regfill

def convert(x):
    if x >= 0.5:
        return 1
    return 0

path = 'G:\BigDataEvent\血糖预测\\'
# test = pd.read_csv(path + 'f_test_a_20180204.csv',encoding='gb2312')
# train = pd.read_csv(path + 'f_train_20180204.csv', encoding='gb2312')

#train_data, test_data = process_eigen.process_eigen(train, test)
train_data = pd.read_csv(path + 'process_train_fb.csv', encoding='gb2312')
test_data = pd.read_csv(path + 'process_test_fb.csv', encoding='gb2312')
print(train_data.info())
print('开始训练。。。')
predictors = [f for f in test_data.columns if f not in ['label', 'id']]

train_preds = np.zeros(train_data.shape[0])
test_preds = np.zeros((test_data.shape[0], 5))
kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=50)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_data.iloc[train_index]
    valid_feat = train_data.iloc[test_index]
    estimator = RandomForestClassifier(n_estimators=500,
                                       min_samples_split=70,
                                       max_depth=12,
                                       min_samples_leaf=10,
                                       max_features=16,
                                       random_state=10,
                                       oob_score=True
                                       )
    estimator.fit(train_feat[predictors], train_feat['label'])
    train_preds[test_index] += estimator.predict(valid_feat[predictors])
    print(estimator.oob_score_)
    test_preds[:,i] = estimator.predict(test_data[predictors])

for i in range(len(predictors)):
    print(i, ':', predictors[i])
feat_imp = pd.Series(estimator.feature_importances_, index=predictors).sort_values(ascending=False)
feat_imp.to_csv('feat_imp_new.csv')
print(len(predictors))
print('线下得分：f1', f1_score(train_data['label'], train_preds))
print('线下得分：precision', precision_score(train_data['label'], train_preds))
print('线下得分：recall', recall_score(train_data['label'], train_preds))
# submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
# submission = submission['pred'].apply(lambda x : convert(x))
# submission.to_csv(r'G:\machine learning\diabetes\classification\submission\f_0305_rf_{}.csv'.format('2'), header=None,
#                   index=False)