import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

def convert(target, x):
    if x == target:
        return 1
    return 0

def convert_na(x):
    if x is True:
        return 1
    return 0

def have_na(x):
    sum = 0
    ind = list(x.index)
    for i in ind:
        sum += x[i]
    if sum == 1400:
        return False
    return True

def process_nan(X, y, test):
    estimator = GradientBoostingRegressor(learning_rate=0.01,
                                            n_estimators=300,
                                            min_samples_split=50,
                                            max_depth=6,
                                            min_samples_leaf=20,
                                            max_features=18,
                                            random_state=10,
                                            subsample=0.8)
    estimator.fit(X, y)
    predicted = estimator.predict(test)
    return predicted


def processNoneCat(X, y, test):
    estimator = GradientBoostingClassifier(learning_rate=0.01,
                                          n_estimators=300,
                                          min_samples_split=50,
                                          max_depth=6,
                                          min_samples_leaf=20,
                                          max_features=18,
                                          random_state=10,
                                          subsample=0.8)
    estimator.fit(X, y)
    predicted = estimator.predict(test)
    return predicted

def process_eigen(train, test_a, test_aa, test_b):
    test_id = test_b.id.values.copy()
    test_a['label'] = test_aa
    train_data = pd.concat([train, test_a])
    train_id = train_data.id.values.copy()
    data = pd.concat([train_data, test_b])

    # 删除缺失值重要性小的特征
    # del_feat = ['SNP25', 'SNP35', 'SNP42', 'SNP50',
    #             'SNP9', 'SNP30', 'SNP44', 'SNP10',
    #             'SNP33', 'SNP26', 'SNP8']
    # data = data.drop(del_feat, axis=1)

    regression_feat = ['年龄', '孕次', '产次', '身高', '孕前体重','RBP4', '分娩时',
                       '孕前BMI', '收缩压', '舒张压', '糖筛孕周',
                       'VAR00007', 'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG',
                       'HDLC', 'LDLC', 'ApoA1', 'ApoB', 'Lpa', 'hsCRP']

    one_hot_feat = [f for f in data.columns if f not in regression_feat]
    one_hot_feat.remove('id')
    one_hot_feat.remove('label')
    #one_hot_feat.remove('SNP2')
    #one_hot_feat = ['BMI分类', 'DM家族史', 'ACEID', 'SNP34']
    # one_hot_feat = ['BMI分类', 'DM家族史', 'ACEID', 'SNP34', 'SNP37', 'SNP40', 'SNP55',
    #                 'SNP38', 'SNP23', 'SNP54', 'SNP53', 'SNP48', 'SNP31', 'SNP21',
    #                 'SNP22']

    #
    # 对离散值做one-hot encoding处理
    for each in one_hot_feat:
        cnt = data[each].value_counts()
        for i in cnt.index:
            temp_feat = '%s_%s' % (int(i), each)
            data[temp_feat] = data[each].apply(lambda x: convert(i, x))
        if have_na(cnt):
            data['na_%s' % each] = data[each].isnull().apply(lambda x: convert_na(x))

    use_to_pred = [f for f in data.columns if f not in ['id', 'label']]
    all_median = data.fillna(data.median(axis=0))
    # 对连续值做回归预测
    for feature in regression_feat:
        print('开始填充{}...'.format(feature))
        temp_feat = [f for f in use_to_pred if f not in [feature]]
        known = data[data[feature].notnull()].id
        unknown = data[data[feature].isnull()].id

        X = all_median[data.id.isin(known)][temp_feat]
        y = all_median[data.id.isin(known)][feature]

        test = all_median[data.id.isin(unknown)][temp_feat]
        data.loc[data[feature].isnull(), [feature]] = process_nan(X, y, test)

    data = data.fillna(0)
    data['VAR_BMI'] = data['VAR00007'] * data['孕前BMI']
    data['VAR_TG'] = data['VAR00007'] * data['TG']
    data['VAR_hsCRP'] = data['VAR00007'] * data['hsCRP']
    data['VAR_wbc'] = data['VAR00007'] * data['wbc']
    #data['VAR_年龄'] = data['VAR00007'] / data['年龄']
    #data['VAR_分娩时'] = data['VAR00007'] * data['分娩时']
    data['产孕'] = data['产次'] / data ['孕次']
    data['产次_年龄'] = data['产次'] / data['年龄']
    data['孕次_年龄'] = data['孕次'] / data['年龄']
    data['身高_体重'] = data['身高'] / data['孕前体重']
    data['收缩压_舒张压'] = data['收缩压'] / data['舒张压']
    data['收缩压__舒张压'] = data['收缩压'] + data['舒张压']
    data['HDLC_LDLC'] = data['HDLC'] / data['LDLC']
    #data['HDLC__LDLC'] = data['HDLC'] + data['LDLC']
    data['ApoA1_ApoB'] = data['ApoA1'] / data['ApoB']
    #data['ApoA1__ApoB'] = data['ApoA1'] + data['ApoB']
    data['分娩时_糖筛孕周'] = data['分娩时'] / data['糖筛孕周']
    #data['分娩时__糖筛孕周'] = data['分娩时'] * data['糖筛孕周']


    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

if __name__ == '__main__':
    path = 'G:\BigDataEvent\血糖预测\\'
    train = pd.read_csv(path + 'f_train_20180204.csv', encoding='gb2312')
    test_a = pd.read_csv(path + 'f_test_a_20180204.csv', encoding='gb2312')
    test_aa = pd.read_csv(path  + 'f_answer_a_20180306.csv', encoding='gb2312',header=None)
    test_b = pd.read_csv(path + 'f_test_b_20180305.csv', encoding='gb2312')

    #print(test_aa)
    train_feat, test_feat = process_eigen(train, test_a, test_aa, test_b)
    #print(train_feat['Unnamed: 0'])
    print(test_feat.columns)
    train_feat.to_csv(path + 'process_train_fb.csv', index=None)
    test_feat.to_csv(path + 'process_test_fb.csv', index=None)

