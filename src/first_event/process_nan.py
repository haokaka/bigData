from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import numpy as np
from dateutil.parser import parse

def convert(x, s, l):
    if x >= s and x < l:
        return 1
    else:
        return 0

data_path = 'G:\BigDataEvent\血糖预测\\'   #存放训练数据和b榜的测试数据的路径，训练数据包括初赛一阶段的训练数据和a榜的测试数据的路径
train_o = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test_a = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')
test_a_answer = pd.read_csv(data_path + 'd_answer_a_20180128.csv', encoding='gb2312',header=None)
test_b = pd.read_csv(data_path + 'd_test_B_20180128.csv', encoding='gb2312')
test_a['血糖'] = test_a_answer

# print(test_a.head(10))
train = pd.concat([train_o, test_a])
train = train[train['血糖'] < 15]  #去除训练数据中的异常值样例，这里的异常值为血糖大于等于15的样例
# print(train.shape)

test_id = test_b.id.values.copy()
train_id = train.id.values.copy()

data = pd.concat([train, test_b])

# data = data[data['shoger'] < 15]
# print('data shape', data.shape)
# data['男'] = data['性别'].map({'男': 1, '女': 0})
# data['女'] = data['性别'].map({'男': 0, '女': 1})
# data = data.drop(['性别'], axis=1)
# data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

del_feature = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体','乙肝核心抗体']
not_pred = ['id', '血糖', '性别', '年龄','体检日期',
            '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
pred_f_list = [f for f in data.columns if f not in not_pred]
use_to_pred = [f for f in data.columns if f not in del_feature]


#初步处理缺失值，先用中位数填充，并且添加了一系列特征，处理好的特征在train_feat和test_feat文件中
median_trian = pd.read_csv('train_feat.csv', encoding='gb2312')
median_test = pd.read_csv('test_feat.csv', encoding='gb2312')
median_all = pd.concat([median_trian, median_test])
# print('median_all shape', median_all.shape)

#删除乙肝相关的特征，因为缺失值太多
#再次处理缺失值，对于某一个特征，将其设置为label，在该特征上没有缺失的样例作为训练数据，使用回归预测的结果填充缺失值
def process_nan(X, y, test):
    estimator = GradientBoostingRegressor(learning_rate=0.005,
                                          n_estimators=2000,
                                          min_samples_split=70,
                                          max_depth=6,
                                          min_samples_leaf=30,
                                          max_features=18,
                                          random_state=10,
                                          subsample=0.8)
    estimator.fit(X, y)
    predicted = estimator.predict(test)
    return predicted



for feature in pred_f_list:
    print('开始填充{}...'.format(feature))
    temp_feat = [f for f in use_to_pred if f not in [feature]]
    known = data[data[feature].notnull()].id
    unknown = data[data[feature].isnull()].id

    X = median_all[median_all.id.isin(known)][temp_feat]
    y = median_all[median_all.id.isin(known)][feature]

    test = median_all[median_all.id.isin(unknown)][temp_feat]
    data.loc[data[feature].isnull(), [feature]] = process_nan(X, y, test)
    # data[[feature], [data[feature].isnull()]] = process_nan(X, y, test)
    # print(feature, 'done..')

# 向使用回归预测填充的缺失值的特征中，重新加入第一步处理缺失值的那些特征
data['天丙'] = (data['*天门冬氨酸氨基转换酶'] / data['*丙氨酸氨基转换酶'])[data['*丙氨酸氨基转换酶'] != 0]
data['天碱'] = (data['*天门冬氨酸氨基转换酶'] / data['*碱性磷酸酶'])[data['*碱性磷酸酶'] != 0]
data['天谷'] = (data['*天门冬氨酸氨基转换酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]
data['丙碱'] = (data['*丙氨酸氨基转换酶'] / data['*碱性磷酸酶'])[data['*碱性磷酸酶'] != 0]
data['丙谷'] = (data['*丙氨酸氨基转换酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]
data['碱谷'] = (data['*碱性磷酸酶'] / data['*r-谷氨酰基转换酶'])[data['*r-谷氨酰基转换酶'] != 0]

    # 加入胆固醇的高低浓度之比
data['胆固醇比'] =(data['高密度脂蛋白胆固醇'] / data['低密度脂蛋白胆固醇'])[data['低密度脂蛋白胆固醇'] != 0]

    # 加入血细胞的各项比
data['白红计数比'] = (data['白细胞计数'] / data['红细胞计数'])[ data['红细胞计数'] != 0]
data['红细胞平均血红蛋白量浓度'] = (data['红细胞平均血红蛋白量'] / data['红细胞平均血红蛋白浓度'])[ data['红细胞平均血红蛋白浓度'] != 0]

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
data['男'] = data['性别'].map({'男': 1, '女': 0})
data['女'] = data['性别'].map({'男': 0, '女': 1})
data['age0'] = data['年龄'].apply(lambda x:convert(x, 0, 29))
data['age1'] = data['年龄'].apply(lambda x: convert(x, 29, 33))
data['age2'] = data['年龄'].apply(lambda x: convert(x, 33, 36))
data['age3'] = data['年龄'].apply(lambda x: convert(x, 36, 40))
data['age4'] = data['年龄'].apply(lambda x: convert(x, 40, 44))
data['age5'] = data['年龄'].apply(lambda x: convert(x, 44, 47))
data['age6'] = data['年龄'].apply(lambda x: convert(x, 47, 51))
data['age7'] = data['年龄'].apply(lambda x: convert(x, 51, 55))
data['age8'] = data['年龄'].apply(lambda x: convert(x, 55, 61))
data['age9'] = data['年龄'].apply(lambda x: convert(x, 61, 77))
data['age9'] = data['年龄'].apply(lambda x: convert(x, 61, 77))
data['age10'] = data['年龄'].apply(lambda x: convert(x, 77, 90))

data['性别'] = data['性别'].map({'男': 1, '女': 0})
data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

# 再次删除id，性别，和乙肝相关数据
del_feat = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
data = data.fillna(data.median(axis=0))
data = data.drop(del_feat, axis=1)

train_data = data[data.id.isin(train_id)]
test_data = data[data.id.isin(test_id)]

predictors = [f for f in test_data.columns if f not in ['血糖','id']]


def my_loss(ground_truth, predictions):
    score = mean_squared_error(ground_truth, predictions) * 0.5
    return score

loss_f = make_scorer(my_loss, greater_is_better=False)


print('开始CV 10折训练...')
scores = []
train_preds = np.zeros(train_data.shape[0])
test_preds = np.zeros((test_data.shape[0], 10))
kf = KFold(len(train_data), n_folds=10, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_data.iloc[train_index]
    valid_feat = train_data.iloc[test_index]
    estimator = GradientBoostingRegressor(learning_rate=0.005,
                                          n_estimators=2000,
                                          min_samples_split=70,
                                          max_depth=6,
                                          min_samples_leaf=30,
                                          max_features=18,
                                          random_state=10,
                                          subsample=0.8)
    estimator.fit(train_feat[predictors], train_feat['血糖'])
    train_preds[test_index] += estimator.predict(valid_feat[predictors])
    test_preds[:,i] = estimator.predict(test_data[predictors])

print('线下得分：    {}'.format(mean_squared_error(train_data['血糖'], train_preds) * 0.5))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'final_test_submission.csv', header=None,
                  index=False, float_format='%.4f')
