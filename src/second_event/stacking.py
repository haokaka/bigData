import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
import xgboost as xgb

def convert(x):
    if x >= 0.5:
        return 1
    return 0

def cout(x):
    res = []
    for each in x:
        res.append(each[1])
    return res

path = 'G:\BigDataEvent\血糖预测\\'
train_data = pd.read_csv(path + 'process_train_fb.csv', encoding='gb2312')
test_data = pd.read_csv(path + 'process_test_fb.csv', encoding='gb2312')
predictors = [f for f in test_data.columns if f not in ['label', 'id']]

params_xgb={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':9,
    'lambda':10,
    'gamma':0.15,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'min_child_weight':1,
    'eta': 0.1,
    'seed':0,
    'nthread':8,
     'silent':1}

rf = RandomForestClassifier(n_estimators=480,
                                           min_samples_split=40,
                                           max_depth=10,
                                           min_samples_leaf=10,
                                           max_features=16,
                                           random_state=10,
                                           oob_score=True
                                           )

gbdt = GradientBoostingClassifier(learning_rate=0.01,
                                            n_estimators=600,
                                            min_samples_split=50,
                                            max_depth=6,
                                            min_samples_leaf=20,
                                            max_features=18,
                                            random_state=10,
                                            subsample=0.8
                                          )

def get_oof(train, test, _xgb, clf=None):
    oof_train = np.zeros((train.shape[0]))
    oof_test = np.zeros((test.shape[0], 5))
    kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=520)

    for i, (train_index, test_index) in enumerate(kf):
        print(i, 'kf...')
        train_X = train.iloc[train_index]
        valid_X = train.iloc[test_index]

        if _xgb == True:
            train_X = xgb.DMatrix(train_X[predictors], label=train_X['label'])
            valid_X = xgb.DMatrix(valid_X[predictors])
            xgb_test = xgb.DMatrix(test[predictors])
            watchlist = [(train_X, 'train')]

            estimator = xgb.train(params_xgb, train_X, num_boost_round=250, evals=watchlist)

            oof_train[test_index] = estimator.predict(valid_X)
            oof_test[:, i] = estimator.predict(xgb_test)
        else:
            xTrain = train_X[predictors]
            yTrain = train_X['label']
            xValid = valid_X[predictors]
            xTest = test[predictors]

            clf.fit(xTrain, yTrain)
            # print(estimator.classes_)
            oof_train[test_index] = cout(clf.predict_proba(xValid))
            oof_test[:, i] = cout(clf.predict_proba(xTest))

    return oof_train.reshape(-1, 1), oof_test.mean(axis=1).reshape(-1, 1)

first_xgb_train, first_xgb_test = get_oof(train_data, test_data, True)
first_rf_train, first_rf_test = get_oof(train_data, test_data, False, rf)
#first_gbdt_train, first_gbdt_test = get_oof(train_data, test_data, False, gbdt)

#first_train_X = pd.DataFrame(np.zeros((first_rf_train.shape[0], 3)), columns=['xgb', 'rf', 'gbdt'])
first_train_X = pd.DataFrame(np.zeros((first_rf_train.shape[0], 2)), columns=['xgb', 'rf'])
first_train_y = train_data['label']
#first_test_X = first_train = pd.DataFrame(np.zeros((first_rf_test.shape[0], 3)), columns=['xgb', 'rf', 'gbdt'])
first_test_X = first_train = pd.DataFrame(np.zeros((first_rf_test.shape[0], 2)), columns=['xgb', 'rf'])
first_train_X['xgb'] = first_xgb_train
first_train_X['rf'] = first_rf_train
#first_train_X['gbdt'] = first_gbdt_train

first_test_X['xgb'] = first_xgb_test
first_test_X['rf'] = first_rf_test
#first_test_X['gbdt'] = first_gbdt_test
final_pred = np.zeros(first_rf_test.shape[0])

clf = linear_model.LinearRegression()
#clf = linear_model.LogisticRegression()
clf.fit(first_train_X, first_train_y)
final_pred[:] = clf.predict(first_test_X)
print(final_pred)
submission = pd.DataFrame({'pred': final_pred})
submission = submission['pred'].apply(lambda x : convert(x))
submission.to_csv(r'G:\machine learning\diabetes\classification\submission\stack2_log_final_result2.csv', header=None,
                  index=False)
