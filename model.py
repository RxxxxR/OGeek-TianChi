# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:23:40 2018

@author: 20277
"""
#from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
#import numpy as np
#import csv

#train_data = pd.read_table("./oppo_round1_train_20180929/oppo_round1_train_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
#vali_data = pd.read_table("./oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
#testA_data = pd.read_table("./oppo_round1_test_A_20180929/oppo_round1_test_A_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
#testB_data = pd.read_table("./oppo_round1_test_B_20181106.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)


#x_data_tag_all = pd.read_csv("./x_data.csv")
#x_vali_tag_all = pd.read_csv("./x_vali.csv")


data_tag = [         
        #######新数据############ 68.4
        "len_query_prediction",#预测序列的长度
        "query_prediction_title_rate",#文章标题在预测序列中的概率
        "max_query_prediction",#预测序列的概率最大值
        "sum_query_prediction",#预测序列的概率总和
        "mean_query_prediction",#预测序列的概率均值
        "var_query_prediction",#预测序列的概率方差
        "max_query_prediction_rate",#新的预测序列的概率最大值与预测序列的概率最大值的比值
        "sum_query_prediction_rate",#新的预测序列的概率最总和与预测序列的概率最总和的比值
        "mean_query_prediction_rate",#新的预测序列的概率最均值与预测序列的概率最均值的比值
        "len_title",#文章标题的长度
        "prefix_title_rate",#前缀词出现在文章标题的长度比值
        "prefix_title_distance_rate",#前缀词与文章标题的编辑距离除以他们的总长度
    
        ##########svd############## 
        'prefix_title_dot',
        'prefix_title_consin',
        'prefix_title_abs_mean',
        'prefix_title_abs_sum',
        'prefix_title_abs_std',
        'prefix_title_abs_max',
        'prefix_title_abs_min',
        'prefix_title_mul_mean',
        'prefix_title_mul_sum',
        'prefix_title_mul_std',
        'prefix_title_mul_max',
        'prefix_title_mul_min',
        'prefix_max_query_dot',
        'prefix_max_query_consin',
        'prefix_max_query_abs_mean',
        'prefix_max_query_abs_sum',
        'prefix_max_query_abs_std',
        'prefix_max_query_abs_max',
        'prefix_max_query_abs_min',
        'prefix_max_query_mul_mean',
        'prefix_max_query_mul_sum',
        'prefix_max_query_mul_std',
        'prefix_max_query_mul_max',
        'prefix_max_query_mul_min',
        'max_query_title_dot',
        'max_query_title_consin',
        'max_query_title_abs_mean',
        'max_query_title_abs_sum',
        'max_query_title_abs_std',
        'max_query_title_abs_max',
        'max_query_title_abs_min',
        'max_query_title_mul_mean',
        'max_query_title_mul_sum',
        'max_query_title_mul_std',
        'max_query_title_mul_max',
        'max_query_title_mul_min', 
        "max_query_title_rate",
        "max_query_title_distance_rate",
        
        ########历史数据统计#########
        "new_max_query_ctr",#预测序列最大值被点击的概率
        "new_title_ctr",#文章标题被点击的概率
        "new_tag_ctr",#文章标签被点击的概率
        "new_tag_len_title_ctr",#文章标题的长度与文章标签被点击的概率       
        "new_tag_len_prefix_ctr",#前缀词的长度与文章标签被点击的概率   
        "new_tag_len_prefix_len_title_ctr",#前缀词的长度与文章标题的长度与文章标签被点击的概率
        "new_tag_title_ctr",#文章标题与文章标签被点击的概率  

    ]

def get_clean_data(x_data):   
    x_data["new_max_query_ctr"] = x_data["max_query_ctr"]
    x_data.loc[x_data["max_query_count"]<=3, "new_max_query_ctr"] = -1
    x_data["new_tag_ctr"] = x_data["tag_ctr"]
    x_data.loc[x_data["tag_count"]<=3, "new_tag_ctr"] = -1
    x_data["new_tag_title_ctr"] = x_data["tag_title_ctr"]
    x_data.loc[x_data["tag_title_count"]<=2, "new_tag_title_ctr"] = -1
    x_data["new_title_ctr"] = x_data["title_ctr"]
    x_data.loc[x_data["title_count"]<=5, "new_title_ctr"] = -1
    x_data["new_tag_len_title_ctr"] = x_data["tag_len_title_ctr"]
    x_data.loc[x_data["tag_len_title_count"]<=3, "new_tag_len_title_ctr"] = -1
    x_data["new_tag_len_prefix_ctr"] = x_data["tag_len_prefix_ctr"]
    x_data.loc[x_data["tag_len_prefix_count"]<=3, "new_tag_len_prefix_ctr"] = -1
    x_data["new_tag_len_prefix_len_title_ctr"] = x_data["tag_len_prefix_len_title_ctr"]
    x_data.loc[x_data["tag_len_prefix_len_title_count"]<=3, "new_tag_len_prefix_len_title_ctr"] = -1
    return x_data



def RandomForestModel(x_data_tag_all, x_vali_tag_all, x_testB_tag_all):
    x_data_tag_all = get_clean_data(x_data_tag_all)
    x_data = x_data_tag_all[data_tag].fillna(-1)
    y_data = x_data_tag_all["label"]
    
    ###################################
    x_vali_tag_all = get_clean_data(x_vali_tag_all)
    x_vali = x_vali_tag_all[data_tag].fillna(-1)
    y_vali = x_vali_tag_all["label"]
    
    x_test_tag_all = get_clean_data(x_testB_tag_all)
    x_test = x_test_tag_all[data_tag].fillna(-1)
    
    
    x_test = pd.concat([x_vali, x_test], axis = 0)
    
    
    from sklearn.model_selection import StratifiedKFold

    n_folds = 5
    seed = 100
    
    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kfold = kfolder.split(x_data, y_data)
    
    RandomForest = RandomForestClassifier(random_state = seed, 
                                          min_samples_split = 3, 
                                          n_estimators = 100,
                                          verbose = 1,
                                          class_weight = "balanced",
                                          n_jobs = 100)#进程数       
    
    preds_list = []
    #(8)72.948
    for train_index, test_index in kfold:
        k_x_train = x_data.iloc[train_index]
        k_y_train = y_data.iloc[train_index]
        k_x_test = x_data.iloc[test_index]
        k_y_test = y_data.iloc[test_index]
        RandomForest.fit(k_x_train, k_y_train)
        imp = RandomForest.feature_importances_
        print(imp)
        y_pred = RandomForest.predict(k_x_test)
        print("f1_score0:",f1_score(k_y_test, y_pred))
        preds = RandomForest.predict(x_test)
        preds_list.append(preds)
    
    preds_df = pd.DataFrame(preds_list).T
    preds_df["mean"] = preds_df.mean(axis=1)
    
    
    threshold = 0.5
    print(threshold)
    preds_df["label"] = preds_df["mean"].apply(lambda item: 1 if item >= threshold else 0)
    vali_preds = preds_df[:x_vali.shape[0]]
    test_preds = preds_df[x_vali.shape[0]:]
    print("f1_score0:", f1_score(y_vali, vali_preds["label"]))
    pd.DataFrame(test_preds).to_csv("./result.csv", index = False, header = False)
    
#    #RandomForest = RandomForestClassifier(max_depth=100, random_state=100, n_estimators= 30)
#    #n_estimators:树的数量, 越多越好，但是性能越差，推荐100
#    #max_depth：数的最大深度，不设置可以达到最高点
#    #oob_score_：是否使用袋外样本
#    #min_samples_split: 设置分支最小样本数，防止过拟合, 数量为3比较合理, 脏数据一般只含1个
#    
#    
#    #（特征选择）树的数量设置为默认值，最终模型树的数量用100
#    #seed = 100
#    #X_train,X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2,random_state = seed)
#    RandomForest = RandomForestClassifier(random_state = 100, 
#                                          min_samples_split = 3, 
#                                          n_estimators = 100,
#                                          oob_score = True,
#                                          verbose = 2,
#                                          class_weight = "balanced",
#                                          n_jobs = 10)#进程数         
#    
#                            
#    
#    #print("交叉验证：")
#    #RandomForest.fit(X_train, y_train)
#    #imp = RandomForest.feature_importances_
#    #print(imp)
#    #print(np.min(imp))
#    #print(data_tag[np.where(imp==np.min(imp))[0][0]])
#    #print("oob_score:",RandomForest.oob_score_)
#    #y_pred = RandomForest.predict(X_test)
#    #print("f1_score0:",f1_score(y_test, y_pred))
#    
#    print("验证集：")
#    RandomForest.fit(x_data, y_data)
#    imp = RandomForest.feature_importances_
#    print(imp)
#    print(np.min(imp))
#    print(data_tag[np.where(imp==np.min(imp))[0][0]])
#    print("oob_score:",RandomForest.oob_score_)
#    y_pred = RandomForest.predict(x_test)
#    #print("f1_score0:",f1_score(y_vali, y_pred))
#    pd.DataFrame(y_pred).to_csv("./result.csv", index = False, header = False)



#vali_data = pd.read_table("./oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
#Y = pd.concat([vali_data, pd.DataFrame(y_pred)], axis = 1)
#Y_True = Y[Y["label"]==Y[0]]
#Y_False = Y[Y["label"]!=Y[0]]
#pd.DataFrame(Y_True).to_csv("./Y_True.csv", index = False, header = True, encoding="utf_8_sig")
#pd.DataFrame(Y_False).to_csv("./Y_False.csv", index = False, header = True, encoding="utf_8_sig")

#导出预测结果
#pd.DataFrame(y_pred).to_csv("./2.csv", index = False, header = False)
