# -*- coding: utf-8 -*-

'prepare training data'

__author__ = 'Junhao He'

import pandas as pd
from dateutil.parser import parse

def convert1(x):
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

def convert(x, s, l):
    if x >= s and x < l:
        return 1
    else:
        return 0

def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])

    # data = data.fillna(data.median(axis=0))
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
    data['age'] = data['年龄'].apply(lambda x: convert1(x))
    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    # data['男'] = data['性别'].map({'男': 1, '女': 0})
    # data['女'] = data['性别'].map({'男': 0, '女': 1})
    # data['age0'] = data['年龄'].apply(lambda x: convert(x, 0, 29))
    # data['age1'] = data['年龄'].apply(lambda x: convert(x, 29, 33))
    # data['age2'] = data['年龄'].apply(lambda x: convert(x, 33, 36))
    # data['age3'] = data['年龄'].apply(lambda x: convert(x, 36, 40))
    # data['age4'] = data['年龄'].apply(lambda x: convert(x, 40, 44))
    # data['age5'] = data['年龄'].apply(lambda x: convert(x, 44, 47))
    # data['age6'] = data['年龄'].apply(lambda x: convert(x, 47, 51))
    # data['age7'] = data['年龄'].apply(lambda x: convert(x, 51, 55))
    # data['age8'] = data['年龄'].apply(lambda x: convert(x, 55, 61))
    # data['age9'] = data['年龄'].apply(lambda x: convert(x, 61, 77))
    # data['age9'] = data['年龄'].apply(lambda x: convert(x, 61, 77))
    # data['age10'] = data['年龄'].apply(lambda x: convert(x, 77, 90))
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

    #删除id，性别，和乙肝相关数据
    del_feat = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
    data = data.drop(del_feat, axis=1)
    data = data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat


def make_feat_one_hot(train, test):
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

    # 删除id，性别，和乙肝相关数据
    del_feat = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
    data = data.fillna(data.median(axis=0))
    data = data.drop(del_feat, axis=1)
    print('before shape:', data.shape)
    data = data[data['shoger'] < 15]
    print('after drop:', data.shape)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

if __name__ == '__main__':
    data_path = 'G:\BigDataEvent\血糖预测\\'
    train_o = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
    test_a = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')
    test_a_answer = pd.read_csv(data_path + 'd_answer_a_20180128.csv', encoding='gb2312')
    test_b = pd.read_csv(data_path + 'd_test_B_20180128.csv', encoding='gb2312')

    test_a['shoger'] = test_a_answer
    train = pd.concat([train_o, test_a])

    train_data, test_data = make_feat_one_hot(train, test_b)
    train_data.to_csv('train_feat.csv')
    test_data.to_csv('test_feat.csv')
    # print('train info:')
    # print(train.info())
    # print('test data info:')
    # print(test_b.info())
