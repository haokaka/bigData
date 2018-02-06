import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

data_path = 'G:\BigDataEvent\血糖预测\\'

train_o = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test_a = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')
test_a_answer = pd.read_csv(data_path + 'd_answer_a_20180128.csv', encoding='gb2312')
test_b = pd.read_csv(data_path + 'd_test_B_20180128.csv', encoding='gb2312')

test_a['shoger'] = test_a_answer
train = pd.concat([train_o, test_a])

print(train['shoger'].head(10))
def convert(x):
    if x < 29:
        return 0
    elif x < 33:
        return 1
    elif x < 36:
        return 2
    elif x < 40:
        return 3
    elif x < 44:
        return 4
    elif x < 47:
        return 5
    elif x < 51:
        return 6
    elif x < 55:
        return 7
    elif x < 61:
        return 8
    elif x < 77:
        return 9
    else:
        return 10

def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])

    data = data.fillna(data.median(axis=0))
    # 加入酶的各项比值
    data['天丙'] = (data['*天门冬氨酸氨基转换酶'] / data['*丙氨酸氨基转换酶'])[data['*丙氨酸氨基转换酶'] != 0]
    data['天碱'] = (data['*天门冬氨酸氨基转换酶'] / data['*碱性磷酸酶'])[data['*碱性磷酸酶'] != 0]
    data['天谷'] = (data['*天门冬氨酸氨基转换酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]
    data['丙碱'] = (data['*丙氨酸氨基转换酶'] / data['*碱性磷酸酶'])[data['*碱性磷酸酶'] != 0]
    data['丙谷'] = (data['*丙氨酸氨基转换酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]
    data['碱谷'] = (data['*碱性磷酸酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]

    # 加入胆固醇的高低浓度之比
    data['胆固醇比'] = (data['高密度脂蛋白胆固醇'] / data['低密度脂蛋白胆固醇'])[data['低密度脂蛋白胆固醇'] != 0]

    # 加入血细胞的各项比
    data['白红计数比'] = (data['白细胞计数'] / data['红细胞计数'])[data['红细胞计数'] != 0]
    data['红细胞平均血红蛋白量浓度'] = (data['红细胞平均血红蛋白量'] / data['红细胞平均血红蛋白浓度'])[data['红细胞平均血红蛋白浓度'] != 0]

    # 加入血小板特征
    data['计数乘平均体积'] = (data['血小板计数'] / data['血小板平均体积'])[data['血小板平均体积'] != 0]
    data['体积比宽度'] = (data['血小板平均体积'] / data['血小板体积分布宽度'])[data['血小板体积分布宽度'] != 0]

    # 加入%细胞
    data['中淋'] = (data['中性粒细胞%'] / data['淋巴细胞%'])[data['淋巴细胞%'] != 0]
    data['中单'] = (data['中性粒细胞%'] / data['单核细胞%'])[data['单核细胞%'] != 0]
    data['淋单'] = (data['淋巴细胞%'] / data['单核细胞%'])[data['单核细胞%'] != 0]
    # data['酸碱'] = (data['嗜酸细胞%'] / data['嗜碱细胞%'])[data['嗜碱细胞%'] != 0]
    # data['酸碱'] = data['酸碱'].apply(lambda x:convert(x))
    # 对性别做one-hot-encode
    # data['男'] = data['性别'].map({'男': 1, '女': 0})
    # data['女'] = data['性别'].map({'男': 0, '女': 1})
    data['age'] = data['年龄'].apply(lambda x: convert(x))
    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

    # 删除id，性别，和乙肝相关数据
    # del_feat = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']

    data = data.fillna(data.median(axis=0))
    # data = data.drop(del_feat, axis=1)

    # data['性别'] = data['性别'].map({'男': 1, '女': 0})
    # data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

    # data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat


train_feat, test_feat = make_feat(train, test_b)

# print(train_feat.head(10))
predictors = [f for f in test_feat.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)


print('开始训练...')
params = {
    'learning_rate': 0.005,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 55,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 10))
kf = KFold(len(train_feat), n_folds=10, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['shoger'], categorical_feature=['性别', 'age'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['shoger'], categorical_feature=['性别', 'age'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])
    print(feat_imp)
print('线下得分：    {}'.format(mean_squared_error(train_feat['shoger'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub_b_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')

