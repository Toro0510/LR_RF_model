# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:37:20 2019

@author: jizeyuan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.externals import joblib
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


def make_var_list(df):
    var_list = []
    for i in df.columns:
        if i.find('var') == 0:
            var_list.append(i)
    return var_list


def model_report(model, model_target, model_predict_result, model_predict_proba):
    print(metrics.classification_report(model_target, model_predict_result))
    fpr_test_tr, tpr_test_tr, th_test_tr = metrics.roc_curve(model_target, model_predict_proba, pos_label=1)
    plt.figure(figsize=[6, 6])
    plt.plot(fpr_test_tr, tpr_test_tr, color="red")
    plt.title('ROC curve')
    print('model_AUC:', metrics.roc_auc_score(model_target, model_predict_proba))
    test_ks = max(tpr_test_tr - fpr_test_tr)
    print('model_KS:', test_ks)


def smote_sample(model_variable, model_target, rate):
    smo = SMOTE(sampling_strategy=rate, random_state=0)
    model_variable_smo, model_target_smo = smo.fit_sample(model_variable, model_target)
    return model_variable_smo, model_target_smo


def calcWOE(var, target):
    gbi = pd.crosstab(var, target)
    gb = target.value_counts()
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    # gbri['iv']=(gbri[1]-gbri[0])*gbri['woe']
    return gbri['woe']


def calcIV(var, target):
    gbi = pd.crosstab(var, target)
    gb = target.value_counts()
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    gbri['iv'] = (gbri[1] - gbri[0]) * gbri['woe']
    return gbri['iv']


def score_exchange_woe(dict, columns_name, value):
    return dict[columns_name][value]


def get_score(coef, woe, p):
    return round(coef * woe * -p, 0)


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


# 读取训练集&跨期样本计算PSI
df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\fraud_model_200605_train.xlsx')
df2 = pd.read_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\fraud_model_200605_oot.xlsx')
model_target = df['target']
model_variable = df[make_var_list(df)]

# 分割样本
train_data, test_data, train_target, test_target = train_test_split(model_variable, model_target, test_size=0.2,
                                                                    random_state=0)
print(len(train_data), len(test_data))


df_temp_train = pd.concat([train_target, train_data], axis=1).reset_index().iloc[:, 1:]
df_temp_test = pd.concat([test_target, test_data], axis=1).reset_index().iloc[:, 1:]
# df_temp_train.to_excel(r'C:\Users\jizeyuan\Desktop\欺诈模型_191122\fraud_model_var_191125_train_temp.xlsx',index=False)
# df_temp_test.to_excel(r'C:\Users\jizeyuan\Desktop\欺诈模型_191122\fraud_model_var_191125_oot__.xlsx',index=False)

# SMOTE_train样本
model_variable_smo, model_target_smo = smote_sample(train_data, train_target, 0.425)
df_smo_target = pd.DataFrame(model_target_smo, columns=['target'])
df_smo_variable = pd.DataFrame(model_variable_smo, columns=make_var_list(df))
print(sum(df_smo_target['target']), len(df_smo_target), sum(df_smo_target['target']) / len(df_smo_target))


# 计算VIF
vif_df = pd.DataFrame()
vif_df["VIF Factor"] = [variance_inflation_factor(train_data.values, i) for i in range(train_data.shape[1])]
vif_df["features"] = train_data.columns


# 计算PSI
psi_df = pd.DataFrame(columns=('var', 'psi'))
for i in train_data:
    for x in train_data[i].unique():
        target_rate = len(train_data[train_data[i] == x]) / len(train_data)
        real_rate = len(df2[df2[i] == x]) / len(df2)
        psi_df = psi_df.append([{'var': i,
                                 'psi': (real_rate - target_rate) * np.log(real_rate / target_rate)}],
                               ignore_index=True)

psi_df = psi_df.groupby('var').sum().reset_index()


# 计算woe
train_var_woe_dict = {}
for i in train_data:
    train_var_woe_dict[i] = dict(calcWOE(df_smo_variable[i], df_smo_target['target']))


# 替换训练集真实值为woe
df_smo_variable_woe = pd.DataFrame()
for i in df_smo_variable.columns:
    df_smo_variable_woe[i] = df_smo_variable[i].apply(lambda x: score_exchange_woe(train_var_woe_dict, i, x))


# 替换测试集真实值为woe
for i in test_data.columns:
    test_data[i] = test_data[i].apply(lambda x: score_exchange_woe(train_var_woe_dict, i, x))


# 随机森林
# best_model=RandomForestClassifier(n_estimators=300,criterion='gini',max_depth=5,min_samples_leaf=30,random_state=10)
# best_model_fit=best_model.fit(train_data, train_target)

# 逻辑回归
best_model = LogisticRegression()
best_model_fit = best_model.fit(df_smo_variable_woe, df_smo_target)
# best_model_fit=best_model.fit(df_smo_variable, df_smo_target)


# train_data_result
model_report(best_model_fit, df_smo_target, best_model_fit.predict(df_smo_variable_woe),
             best_model_fit.predict_proba(df_smo_variable_woe)[:, 1])
model_predict_proba_ = best_model_fit.predict_proba(df_smo_variable_woe)[:, 1]
printKS(df_smo_target, model_predict_proba_)


# test_data_result
model_predict_result = best_model_fit.predict(test_data)
model_predict_proba = best_model_fit.predict_proba(test_data)[:, 1]
model_report(best_model_fit, test_target, model_predict_result, model_predict_proba)
printKS(test_target, model_predict_proba)


# 随机森林变量重要性&逻辑回归系数
# importances=sorted(zip(model_variable.columns,map(lambda x: round(x, 4),best_model_fit.feature_importances_)),reverse=True)
# print(pd.DataFrame(importances,columns=['var','feature_importances']).sort_values(by=['feature_importances'],ascending=False))
coef = sorted(zip(model_variable.columns, map(lambda x: round(x, 4), list(best_model_fit.coef_[0]))), reverse=True)
coef_df = pd.DataFrame(coef, columns=['var', 'coef'])
print(coef_df)
print('intercept_:', best_model_fit.intercept_)


# 计算评分
base_ratio = sum(df_smo_target['target']) / len(df_smo_target)
p = 50 / np.log(2)
q = 600 + p * np.log(base_ratio)
baseScore = round(q - p * best_model_fit.intercept_[0], 0)
print('baseScore: ', baseScore)


woe_list = []
for i in train_data:
    for x in dict(calcWOE(df_smo_variable[i], df_smo_target['target'])):
        coef = coef_df[coef_df['var'] == i]['coef'].values[0]
        vif = vif_df[vif_df['features'] == i]['VIF Factor'].values[0]
        psi = psi_df[psi_df['var'] == i]['psi'].values[0]
        woe_list.append([str(i) + '_' + str(x),
                         dict(calcWOE(df_smo_variable[i], df_smo_target['target']))[x],
                         dict(calcIV(df_smo_variable[i], df_smo_target['target']))[x],
                         coef,
                         vif,
                         psi,
                         get_score(coef, dict(calcWOE(df_smo_variable[i], df_smo_target['target']))[x], p
                                   )])

df_woe = pd.DataFrame(woe_list, columns=['var', 'WOE', 'IV', 'coef', 'vif', 'psi', 'score'])
print(df_woe)


joblib.dump(best_model, r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\model_LR.pkl', compress=3)
df_woe.to_excel(r'C:\Users\jizeyuan\Desktop\反欺诈模型_200605\woe.xlsx', index=False)
