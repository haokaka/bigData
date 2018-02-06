import pandas as pd

'prepare training data'

__author__ = 'Junhao He'


def process_data(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()

    data = pd.concat([train, test])
    data = data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

print('ok')
if __name__ == '__main__':
    path = 'G:\BigDataEvent\血糖预测\\'
    test = pd.read_csv(path + 'f_test_a_20180204.csv',encoding='gb2312')
    train = pd.read_csv(path + 'f_train_20180204.csv', encoding='gb2312')

    train_feat, test_feat = process_data(train, test)
    print(train_feat.info())
    print(test_feat.info())