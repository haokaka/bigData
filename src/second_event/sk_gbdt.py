import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import make_trainData
from sklearn.metrics import make_scorer

def convert(x):
    if x >= 0.5:
        return 1
    return 0

path = 'G:\BigDataEvent\血糖预测\\'
test = pd.read_csv(path + 'f_test_a_20180204.csv',encoding='gb2312')
train = pd.read_csv(path + 'f_train_20180204.csv', encoding='gb2312')

train_data, test_data = make_trainData.process_data(train, test)

print('开始训练。。。')
predictors = [f for f in test_data.columns if f not in ['label', 'id']]

scores = []
train_preds = np.zeros(train_data.shape[0])
test_preds = np.zeros((test_data.shape[0], 10))
kf = KFold(len(train_data), n_folds=10, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_data.iloc[train_index]
    valid_feat = train_data.iloc[test_index]
    estimator = GradientBoostingClassifier(learning_rate=0.01,
                                            n_estimators=600,
                                            min_samples_split=70,
                                            max_depth=6,
                                            min_samples_leaf=20,
                                            max_features=18,
                                            random_state=10,
                                            subsample=0.8
                                          )
    estimator.fit(train_feat[predictors], train_feat['label'])
    train_preds[test_index] += estimator.predict(valid_feat[predictors])
    test_preds[:,i] = estimator.predict(test_data[predictors])

print('线下得分：', f1_score(train_data['label'], train_preds))
submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission = submission['pred'].apply(lambda x : convert(x))
submission.to_csv(r'f_0205_sub_{}.csv'.format('par'), header=None,
                  index=False)