# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:42:30 2019

@author: jizeyuan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.externals import joblib
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


def model_report(model, model_target, model_predict_result, model_predict_proba):
    print(metrics.classification_report(model_target, model_predict_result))
    fpr_test_tr, tpr_test_tr, th_test_tr = metrics.roc_curve(model_target, model_predict_proba, pos_label=1)
    plt.figure(figsize=[6, 6])
    plt.plot(fpr_test_tr, tpr_test_tr, color="red")
    plt.title('ROC curve')
    print('model_AUC:', metrics.roc_auc_score(model_target, model_predict_proba))
    test_ks = max(tpr_test_tr - fpr_test_tr)
    print('model_KS:', test_ks)


def make_var_list(df):
    var_list = []
    for i in df.columns:
        if i.find('var') == 0:
            var_list.append(i)
    return var_list


def make_var_woe_list(df):
    var_list = []
    for i in df.columns:
        if i.find('woe_') == 0:
            var_list.append(i)
    return var_list


def score_exchange(dic_result, columns_name, value):
    dic_columns_name = columns_name + '_' + str(int(value))
    return dic_result[dic_columns_name]


def proba_to_score(proba):
    score = q + p * np.log((1 - proba) / proba)
    return round(score, 0)


def findksIdx(fpr, tpr):
    ks = 0
    ksIdx = 0
    for i in range(fpr.size):
        if tpr[i] - fpr[i] > ks:
            ks = tpr[i] - fpr[i]
            ksIdx = i
    return ksIdx


def printKS(target, proba):
    fpr_myscore, tpr_myscore, _ = metrics.roc_curve(target, proba, pos_label=1)
    ksIdx_myscore = findksIdx(fpr_myscore, tpr_myscore)
    myscore_ks = tpr_myscore[ksIdx_myscore] - fpr_myscore[ksIdx_myscore]
    plt.figure(figsize=[10, 8])
    plt.plot([i / fpr_myscore.size for i in range(fpr_myscore.size)], fpr_myscore, label="ks_myscore", color="r")
    plt.plot([i / tpr_myscore.size for i in range(tpr_myscore.size)], tpr_myscore, label="ks_myscore", color="r")
    # plt.plot([i/fpr_AscoreV2.size for i in range(fpr_AscoreV2.size)],fpr_AscoreV2,label = "ks_AscoreV2",color="b")
    # plt.plot([i/tpr_AscoreV2.size for i in range(tpr_AscoreV2.size)],tpr_AscoreV2,label = "ks_AscoreV2",color="b")
    plt.vlines(x=ksIdx_myscore / fpr_myscore.size, ymin=fpr_myscore[ksIdx_myscore], ymax=tpr_myscore[ksIdx_myscore],
               colors="r")
    # plt.vlines(x=ksIdx_AscoreV2/fpr_AscoreV2.size,ymin=fpr_AscoreV2[ksIdx_AscoreV2],ymax=tpr_AscoreV2[ksIdx_AscoreV2],colors="blue")
    plt.legend(["score_ks = %.4f" % myscore_ks])
    plt.show()


model = joblib.load(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\model_LR.pkl')
df2 = pd.read_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\fraud_model_200605_oot.xlsx')
woe_dict = pd.read_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\woe.xlsx', sheet_name='Sheet1')
woe_dict = dict(pd.Series(data=woe_dict['score'].values, index=woe_dict['var']))
df2.info()

# 将跨期样本数据映射为WOE
columns_list = make_var_list(df2)
for i in columns_list:
    df2['woe_' + i] = df2[i].apply(lambda x: score_exchange_woe(train_var_woe_dict, i, x))


model_target = df2['target']
model_variable = df2[make_var_woe_list(df2)]
model_predict_result = model.predict(model_variable.values)
model_predict_proba = model.predict_proba(model_variable.values)[:, 1]
model_report(model, model_target, model_predict_result, model_predict_proba)
printKS(model_target, model_predict_proba)


# 从预测概率直接计算评分
proba_to_score_list = []
for i in model_predict_proba:
    proba_to_score_list.append(proba_to_score(i))


# 计算变量映射得分后汇总
columns_list = make_var_list(df2)
for i in columns_list:
    model_variable['x_score_' + i] = df2[i].apply(lambda x: score_exchange(woe_dict, i, x))

model_variable_score = model_variable.filter(regex='x_score|trade_no')
model_variable_score['score_total'] = model_variable_score.apply(lambda x: x.sum() + baseScore, axis=1)


predict_result = pd.concat([df2,
                            model_variable_score,
                            pd.DataFrame(model_predict_result, columns=['predict_result']),
                            pd.DataFrame(model_predict_proba, columns=['predict_proba']),
                            pd.DataFrame(proba_to_score_list, columns=['proba_to_score'])],
                           axis=1)
predict_result.to_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\fraud_model_result.xlsx')

