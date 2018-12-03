# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:19:11 2018

@author: 20277
"""
import pandas as pd
import csv
import json
import get_feature
import model
import sys
import jieba

def get_query_prediction_statistics(train_data):
    x_data = train_data
    print("将query_prediction转换成json格式(慢)")
    x_data["query_prediction_json"] = x_data.apply(lambda x:json.loads(str(x["query_prediction"])), axis =1)
    print("获取预测序列中概率最大值")
    x_data["max_query"] = x_data.apply(lambda x:get_feature.get_max_query_prediction_key(x["query_prediction_json"]), axis=1)
    print("预测序列统计值")
    x_data["statistics_query_prediction"] = x_data.apply(lambda x:get_feature.get_statistics_query_prediction(x["query_prediction_json"], x["title"]), axis =1)
    print("分解统计值")
    statistics = x_data["statistics_query_prediction"].str.split(',',expand=True)#多名字分列
    statistics.columns=['max_query_prediction','min_query_prediction','sum_query_prediction','mean_query_prediction','query_prediction_title_rate','len_query_prediction','med_query_prediction','var_query_prediction']
    statistics = statistics.astype("float64")
    x_data=x_data.join(statistics)
    print("statistics_new_query_prediction预测序列统计值")
    x_data["statistics_new_query_prediction"] = x_data.apply(lambda x:get_feature.get_statistics_new_query_prediction(x["query_prediction_json"], x["title"]), axis =1)
    print("分解新的统计值")
    new_statistics = x_data["statistics_new_query_prediction"].str.split(',',expand=True)#多名字分列
    new_statistics.columns=['max_new_query_prediction','min_new_query_prediction','sum_new_query_prediction','mean_new_query_prediction','med_new_query_prediction','var_new_query_prediction']
    new_statistics = new_statistics.astype("float64")  
    x_data=x_data.join(new_statistics)
    print("新的预测序列中概率总值在原预测序列概率总值的占比")
    x_data["sum_query_prediction_rate"] = (x_data["sum_new_query_prediction"]/x_data["sum_query_prediction"]).astype("float64")
    print("新的预测序列中概率最大值在原预测序列概率最大值的占比")
    x_data["max_query_prediction_rate"] = (x_data["max_new_query_prediction"]/x_data["max_query_prediction"]).astype("float64")
    print("新的预测序列中概率均值在原预测序列概率均值的占比")
    x_data["mean_query_prediction_rate"] = (x_data["mean_new_query_prediction"]/x_data["mean_query_prediction"]).astype("float64")
    return x_data

def get_statistics(train_data):
    x_data = pd.DataFrame([])
    print("len_prefix:前缀词长度")
    x_data["len_prefix"] = train_data.apply(lambda x:len(str(x["prefix"])), axis =1)
    print("len_title:文章标题长度")
    x_data["len_title"] = train_data.apply(lambda x:len(str(x["title"])), axis =1) 
    print("len_max_query:预测序列最大值的长度")
    x_data["len_max_query"] = train_data.apply(lambda x:len(str(x["max_query"])), axis =1)
    return x_data

def get_max_query_statistics(train_data, max_query_ctr, max_query_count, len_max_query_ctr, len_max_query_count, len_query_prediction_max_query_ctr, len_query_prediction_max_query_count):
    x_data = pd.DataFrame([])
    print("max_query_title_rate前缀词的长度占文章标题长度的比值")
    x_data["max_query_title_rate"] = train_data.apply(lambda x:get_feature.get_prefix_title_rate(x["max_query"], x["title"]), axis =1)
    print("max_query_title_distance: 前缀词与文章标题的编辑距离")
    x_data["max_query_title_distance"] = train_data.apply(lambda x:get_feature.levenshtein(x["max_query"], x["title"]), axis = 1)
    print("max_query_ctr: max_query被点击的概率")
    x_data["max_query_ctr"] = pd.merge(pd.DataFrame(train_data["max_query"]), max_query_ctr[["max_query","max_query_ctr"]], how="left", on="max_query")["max_query_ctr"]
    print("max_query_count: max_query出现的频率")
    x_data["max_query_count"] = pd.merge(pd.DataFrame(train_data["max_query"]), max_query_count[["max_query","max_query_count"]], how="left", on="max_query")["max_query_count"]
    print("len_max_query_ctr: len_max_query被点击的概率")
    x_data["len_max_query_ctr"] = pd.merge(pd.DataFrame(train_data["len_max_query"]), len_max_query_ctr[["len_max_query","len_max_query_ctr"]], how="left", on="len_max_query")["len_max_query_ctr"]
    print("len_max_query_count: len_max_query出现的频率")
    x_data["len_max_query_count"] = pd.merge(pd.DataFrame(train_data["len_max_query"]), len_max_query_count[["len_max_query","len_max_query_count"]], how="left", on="len_max_query")["len_max_query_count"]
    print("len_query_prediction_max_query_ctr: len_query_prediction_max_query被点击的概率")
    x_data["len_query_prediction_max_query_ctr"] = pd.merge(train_data[["len_query_prediction","max_query"]], len_query_prediction_max_query_ctr[["len_query_prediction","max_query", "len_query_prediction_max_query_ctr"]], how="left", on=["len_query_prediction","max_query"])["len_query_prediction_max_query_ctr"] 
    print("len_query_prediction_max_query_count: len_query_prediction_max_query出现的频率")
    x_data["len_query_prediction_max_query_count"] = pd.merge(train_data[["len_query_prediction","max_query"]], len_query_prediction_max_query_count[["len_query_prediction","max_query", "len_query_prediction_max_query_count"]], how="left", on=["len_query_prediction","max_query"])["len_query_prediction_max_query_count"]
    
    x_data["max_query_title_rate"] = train_data.apply(lambda x:get_feature.get_prefix_title_rate(x["max_query"], x["title"]), axis =1)
    x_data["max_query_title_distance"] = train_data.apply(lambda x:get_feature.levenshtein(x["max_query"], x["title"]), axis = 1)
    x_data["max_query_title_distance_rate"] = x_data["max_query_title_distance"]/(train_data["len_max_query"]+train_data["len_title"]) 

    return x_data

def get_max_query_tag_statistics(train_data, tag_max_query_ctr, tag_max_query_count, tag_len_max_query_ctr, tag_len_max_query_count, title_tag_max_query_ctr, title_tag_max_query_count, tag_len_max_query_len_title_ctr, tag_len_max_query_len_title_count):
    x_data = pd.DataFrame([])
    print("tag_max_query_ctr: tag_max_query被点击的概率")
    x_data["tag_max_query_ctr"] = pd.merge(train_data[["tag","max_query"]], tag_max_query_ctr[["tag","max_query","tag_max_query_ctr"]], how="left", on=["tag","max_query"])["tag_max_query_ctr"]
    print("tag_max_query_count: tag_max_query出现的频率")
    x_data["tag_max_query_count"] = pd.merge(train_data[["tag","max_query"]], tag_max_query_count[["tag","max_query","tag_max_query_count"]], how="left", on=["tag","max_query"])["tag_max_query_count"]
    print("tag_len_max_query_ctr: tag_len_max_query被点击的概率")
    x_data["tag_len_max_query_ctr"] = pd.merge(train_data[["tag","len_max_query"]], tag_len_max_query_ctr[["tag","len_max_query","tag_len_max_query_ctr"]], how="left", on=["tag","len_max_query"])["tag_len_max_query_ctr"]
    print("tag_len_max_query_count: tag_len_max_query出现的频率")
    x_data["tag_len_max_query_count"] = pd.merge(train_data[["tag","len_max_query"]], tag_len_max_query_count[["tag","len_max_query","tag_len_max_query_count"]], how="left", on=["tag","len_max_query"])["tag_len_max_query_count"]
    print("title_tag_max_query_ctr: title_tag_max_query被点击的概率")
    x_data["title_tag_max_query_ctr"] = pd.merge(train_data[["title", "tag", "max_query"]], title_tag_max_query_ctr[["title", "tag", "max_query", "title_tag_max_query_ctr"]], how="left", on=["title", "tag", "max_query"])["title_tag_max_query_ctr"]
    print("title_tag_max_query_count: title_tag_max_query出现的频率")
    x_data["title_tag_max_query_count"] = pd.merge(train_data[["title", "tag", "max_query"]], title_tag_max_query_count[["title", "tag", "max_query", "title_tag_max_query_count"]], how="left", on=["title", "tag", "max_query"])["title_tag_max_query_count"]
    print("tag_len_max_query_len_title_ctr: tag_len_max_query_len_title被点击的概率")
    x_data["tag_len_max_query_len_title_ctr"] = pd.merge(train_data[["tag","len_max_query", "len_title"]], tag_len_max_query_len_title_ctr[["tag","len_max_query", "len_title","tag_len_max_query_len_title_ctr"]], how="left", on=["tag","len_max_query","len_title"])["tag_len_max_query_len_title_ctr"]
    print("tag_len_max_query_len_title_count: tag_len_max_query_len_title出现频率")
    x_data["tag_len_max_query_len_title_count"] = pd.merge(train_data[["tag","len_max_query", "len_title"]], tag_len_max_query_len_title_count[["tag","len_max_query", "len_title","tag_len_max_query_len_title_count"]], how="left", on=["tag","len_max_query","len_title"])["tag_len_max_query_len_title_count"]
    return x_data

def get_prefix_statistics(train_data, prefix_ctr, prefix_count, len_prefix_ctr, len_prefix_count, len_query_prediction_prefix_ctr, len_query_prediction_prefix_count):
    x_data = pd.DataFrame([])
    print("prefix_title_rate前缀词的长度占文章标题长度的比值")
    x_data["prefix_title_rate"] = train_data.apply(lambda x:get_feature.get_prefix_title_rate(x["prefix"], x["title"]), axis =1)
    print("prefix_title_distance: 前缀词与文章标题的编辑距离")
    x_data["prefix_title_distance"] = train_data.apply(lambda x:get_feature.levenshtein(x["prefix"], x["title"]), axis = 1)
    print("prefix_ctr: prefix被点击的概率")
    x_data["prefix_ctr"] = pd.merge(pd.DataFrame(train_data["prefix"]), prefix_ctr[["prefix","prefix_ctr"]], how="left", on="prefix")["prefix_ctr"]
    print("prefix_count: prefix出现的频率")
    x_data["prefix_count"] = pd.merge(pd.DataFrame(train_data["prefix"]), prefix_count[["prefix","prefix_count"]], how="left", on="prefix")["prefix_count"]
    print("len_prefix_ctr: len_prefix被点击的概率")
    x_data["len_prefix_ctr"] = pd.merge(pd.DataFrame(train_data["len_prefix"]), len_prefix_ctr[["len_prefix","len_prefix_ctr"]], how="left", on="len_prefix")["len_prefix_ctr"]
    print("len_prefix_count: len_prefix出现的频率")
    x_data["len_prefix_count"] = pd.merge(pd.DataFrame(train_data["len_prefix"]), len_prefix_count[["len_prefix","len_prefix_count"]], how="left", on="len_prefix")["len_prefix_count"]
    print("len_query_prediction_prefix_ctr: len_query_prediction_prefix被点击的概率")
    x_data["len_query_prediction_prefix_ctr"] = pd.merge(train_data[["len_query_prediction","prefix"]], len_query_prediction_prefix_ctr[["len_query_prediction","prefix", "len_query_prediction_prefix_ctr"]], how="left", on=["len_query_prediction","prefix"])["len_query_prediction_prefix_ctr"] 
    print("len_query_prediction_prefix_count: len_query_prediction_prefix出现的频率")
    x_data["len_query_prediction_prefix_count"] = pd.merge(train_data[["len_query_prediction","prefix"]], len_query_prediction_prefix_count[["len_query_prediction","prefix", "len_query_prediction_prefix_count"]], how="left", on=["len_query_prediction","prefix"])["len_query_prediction_prefix_count"]
    return x_data
    
def get_title_statistics(train_data, title_ctr, title_count, len_title_ctr, len_title_count):
    x_data = pd.DataFrame([])
    print("title_ctr: title被点击的概率")
    x_data["title_ctr"] = pd.merge(pd.DataFrame(train_data["title"]), title_ctr[["title","title_ctr"]], how="left", on="title")["title_ctr"]
    print("title_count: title出现的频率")
    x_data["title_count"] = pd.merge(pd.DataFrame(train_data["title"]), title_count[["title","title_count"]], how="left", on="title")["title_count"]
    print("len_title_ctr: len_title被点击的概率")
    x_data["len_title_ctr"] = pd.merge(pd.DataFrame(train_data["len_title"]), len_title_ctr[["len_title","len_title_ctr"]], how="left", on="len_title")["len_title_ctr"]
    print("len_title_count: len_title出现的频率")
    x_data["len_title_count"] = pd.merge(pd.DataFrame(train_data["len_title"]), len_title_count[["len_title","len_title_count"]], how="left", on="len_title")["len_title_count"]
    print("prefix_title_distance_rate: 前缀词与文章标题的编辑距离处以标题+前缀词的总长度")
    x_data["prefix_title_distance_rate"] = train_data["prefix_title_distance"]/(train_data["len_prefix"]+train_data["len_title"]) 
    return x_data

def get_tag_statistics(train_data, tag_ctr, tag_title_ctr, tag_title_count, tag_len_title_ctr, tag_len_title_count, tag_prefix_ctr, tag_len_prefix_ctr, tag_len_prefix_count, title_tag_prefix_ctr, title_tag_prefix_count, tag_len_prefix_len_title_ctr, tag_len_prefix_len_title_count, tag_len_query_prediction_ctr):
    x_data = pd.DataFrame([])
    print("tag_ctr: tag被点击的概率")
    x_data["tag_ctr"] = pd.merge(pd.DataFrame(train_data["tag"]), tag_ctr[["tag","tag_ctr"]], how="left", on="tag")["tag_ctr"]
    print("tag_count: tag出现的次数")
    x_data["tag_count"] = pd.merge(pd.DataFrame(train_data["tag"]), tag_count[["tag","tag_count"]], how="left", on="tag")["tag_count"]
    print("tag_title_ctr: tag_title被点击的概率")
    x_data["tag_title_ctr"] = pd.merge(train_data[["tag","title"]], tag_title_ctr[["tag","title","tag_title_ctr"]], how="left", on=["tag","title"])["tag_title_ctr"]
    print("tag_title_count: tag_title出现的频率")
    x_data["tag_title_count"] = pd.merge(train_data[["tag","title"]], tag_title_count[["tag","title","tag_title_count"]], how="left", on=["tag","title"])["tag_title_count"]
    print("tag_len_title_ctr: tag_len_title被点击的概率")
    x_data["tag_len_title_ctr"] = pd.merge(train_data[["tag","len_title"]], tag_len_title_ctr[["tag","len_title","tag_len_title_ctr"]], how="left", on=["tag","len_title"])["tag_len_title_ctr"]
    print("tag_len_title_count: tag_len_title出现的频率")
    x_data["tag_len_title_count"] = pd.merge(train_data[["tag","len_title"]], tag_len_title_count[["tag","len_title","tag_len_title_count"]], how="left", on=["tag","len_title"])["tag_len_title_count"]
    print("tag_prefix_ctr: tag_prefix被点击的概率")
    x_data["tag_prefix_ctr"] = pd.merge(train_data[["tag","prefix"]], tag_prefix_ctr[["tag","prefix","tag_prefix_ctr"]], how="left", on=["tag","prefix"])["tag_prefix_ctr"]
    print("tag_len_prefix_ctr: tag_len_prefix被点击的概率")
    x_data["tag_len_prefix_ctr"] = pd.merge(train_data[["tag","len_prefix"]], tag_len_prefix_ctr[["tag","len_prefix","tag_len_prefix_ctr"]], how="left", on=["tag","len_prefix"])["tag_len_prefix_ctr"]
    print("tag_len_prefix_count: tag_len_prefix出现的频率")
    x_data["tag_len_prefix_count"] = pd.merge(train_data[["tag", "len_prefix"]], tag_len_prefix_count[["tag","len_prefix","tag_len_prefix_count"]], how="left", on=["tag","len_prefix"])["tag_len_prefix_count"]
    print("title_tag_prefix_ctr: title_tag_prefix被点击的概率")
    x_data["title_tag_prefix_ctr"] = pd.merge(train_data[["title", "tag", "prefix"]], title_tag_prefix_ctr[["title", "tag", "prefix", "title_tag_prefix_ctr"]], how="left", on=["title", "tag", "prefix"])["title_tag_prefix_ctr"]
    print("title_tag_prefix_count: title_tag_prefix出现的频率")
    x_data["title_tag_prefix_count"] = pd.merge(train_data[["title", "tag", "prefix"]], title_tag_prefix_count[["title", "tag", "prefix", "title_tag_prefix_count"]], how="left", on=["title", "tag", "prefix"])["title_tag_prefix_count"]
    print("tag_len_prefix_len_title_ctr: tag_len_prefix_len_title被点击的概率")
    x_data["tag_len_prefix_len_title_ctr"] = pd.merge(train_data[["tag","len_prefix", "len_title"]], tag_len_prefix_len_title_ctr[["tag","len_prefix", "len_title","tag_len_prefix_len_title_ctr"]], how="left", on=["tag","len_prefix","len_title"])["tag_len_prefix_len_title_ctr"]
    print("tag_len_prefix_len_title_count: tag_len_prefix_len_title出现频率")
    x_data["tag_len_prefix_len_title_count"] = pd.merge(train_data[["tag","len_prefix", "len_title"]], tag_len_prefix_len_title_count[["tag","len_prefix", "len_title","tag_len_prefix_len_title_count"]], how="left", on=["tag","len_prefix","len_title"])["tag_len_prefix_len_title_count"]
    print("tag_len_query_prediction_ctr: tag_len_query_prediction点击的概率")
    x_data["tag_len_query_prediction_ctr"] = pd.merge(train_data[["tag","len_query_prediction"]], tag_len_query_prediction_ctr[["tag","len_query_prediction","tag_len_query_prediction_ctr"]], how="left", on=["tag","len_query_prediction"])["tag_len_query_prediction_ctr"]
    return x_data

def get_list_sentences(train_data):
    data = pd.DataFrame([])
    data['prefix_list'] = train_data['prefix'].map(lambda x : ' '.join(jieba.cut(str(x))))
    data['title_list'] = train_data['title'].map(lambda x : ' '.join(jieba.cut(str(x))))
    data['max_query_list'] = train_data['max_query'].map(lambda x : ' '.join(jieba.cut(str(x))))
    data['prefix_list'].fillna(' ', inplace=True)
    data['title_list'].fillna(' ', inplace=True)
    data['max_query_list'].fillna(' ', inplace=True)
    return data


if __name__=="__main__":
    train = str(sys.argv[1])
    vali = str(sys.argv[2])
    testB = str(sys.argv[3])
#    train = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
#    vali = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'
#    testB = '/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'
    
    print("训练集")
    #train_data = pd.read_table("./oppo_round1_train_20180929/oppo_round1_train_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    train_data = pd.read_table(train, sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    print("预测序列统计值")
    x_data_tag_all = get_query_prediction_statistics(train_data.fillna('{}'))
    train_statistics = get_statistics(x_data_tag_all)
    x_data_tag_all = pd.concat([x_data_tag_all, train_statistics], axis = 1)
    max_query_ctr = x_data_tag_all.groupby(["max_query"], as_index=False)["label"].agg({"max_query_ctr":"mean"})
    max_query_count = x_data_tag_all.groupby(["max_query"], as_index=False)["label"].agg({"max_query_count":"count"})
    len_max_query_ctr = x_data_tag_all.groupby(["len_max_query"], as_index=False)["label"].agg({"len_max_query_ctr":"mean"})
    len_max_query_count = x_data_tag_all.groupby(["len_max_query"], as_index=False)["label"].agg({"len_max_query_count":"count"})
    len_query_prediction_max_query_ctr = x_data_tag_all.groupby(['len_query_prediction', "max_query"], as_index=False)["label"].agg({"len_query_prediction_max_query_ctr":"mean"})
    len_query_prediction_max_query_count = x_data_tag_all.groupby(['len_query_prediction', "max_query"], as_index=False)["label"].agg({"len_query_prediction_max_query_count":"count"})
    train_max_query_statistics = get_max_query_statistics(x_data_tag_all, max_query_ctr, max_query_count, len_max_query_ctr, len_max_query_count, len_query_prediction_max_query_ctr, len_query_prediction_max_query_count)
    x_data_tag_all = pd.concat([x_data_tag_all, train_max_query_statistics], axis = 1)
    print("前缀词统计值")
    prefix_ctr = x_data_tag_all.groupby(["prefix"], as_index=False)["label"].agg({"prefix_ctr":"mean"})
    prefix_count = x_data_tag_all.groupby(["prefix"], as_index=False)["label"].agg({"prefix_count":"count"})
    len_prefix_ctr = x_data_tag_all.groupby(["len_prefix"], as_index=False)["label"].agg({"len_prefix_ctr":"mean"})
    len_prefix_count = x_data_tag_all.groupby(["len_prefix"], as_index=False)["label"].agg({"len_prefix_count":"count"})
    len_query_prediction_prefix_ctr = x_data_tag_all.groupby(['len_query_prediction', "prefix"], as_index=False)["label"].agg({"len_query_prediction_prefix_ctr":"mean"})
    len_query_prediction_prefix_count = x_data_tag_all.groupby(['len_query_prediction', "prefix"], as_index=False)["label"].agg({"len_query_prediction_prefix_count":"count"})
    train_prefix_statistics = get_prefix_statistics(x_data_tag_all, prefix_ctr, prefix_count, len_prefix_ctr, len_prefix_count, len_query_prediction_prefix_ctr, len_query_prediction_prefix_count)
    x_data_tag_all = pd.concat([x_data_tag_all, train_prefix_statistics], axis = 1)
    print("文章标题统计值")
    title_ctr = x_data_tag_all.groupby(["title"], as_index=False)["label"].agg({"title_ctr":"mean"})
    title_count = x_data_tag_all.groupby(["title"], as_index=False)["label"].agg({"title_count":"count"})
    len_title_ctr = x_data_tag_all.groupby(["len_title"], as_index=False)["label"].agg({"len_title_ctr":"mean"})
    len_title_count = x_data_tag_all.groupby(["len_title"], as_index=False)["label"].agg({"len_title_count":"count"})
    train_title_statistics = get_title_statistics(x_data_tag_all, title_ctr, title_count, len_title_ctr, len_title_count)
    x_data_tag_all = pd.concat([x_data_tag_all, train_title_statistics], axis = 1) 
    print("文章标签统计值")
    tag_ctr = x_data_tag_all.groupby(["tag"], as_index=False)["label"].agg({"tag_ctr":"mean"})
    tag_count = x_data_tag_all.groupby(["tag"], as_index=False)["label"].agg({"tag_count":"count"})
    tag_title_ctr = x_data_tag_all.groupby(["tag", "title"], as_index=False)["label"].agg({"tag_title_ctr":"mean"})
    tag_title_count = x_data_tag_all.groupby(["tag", "title"], as_index=False)["label"].agg({"tag_title_count":"count"})
    tag_len_title_ctr = x_data_tag_all.groupby(["tag", "len_title"], as_index=False)["label"].agg({"tag_len_title_ctr":"mean"})
    tag_len_title_count = x_data_tag_all.groupby(["tag", "len_title"], as_index=False)["label"].agg({"tag_len_title_count":"count"})
    tag_prefix_ctr = x_data_tag_all.groupby(["tag", "prefix"], as_index=False)["label"].agg({"tag_prefix_ctr":"mean"})
    tag_len_prefix_ctr = x_data_tag_all.groupby(["tag", "len_prefix"], as_index=False)["label"].agg({"tag_len_prefix_ctr":"mean"})
    tag_len_prefix_count = x_data_tag_all.groupby(["tag", "len_prefix"], as_index=False)["label"].agg({"tag_len_prefix_count":"count"})
    title_tag_prefix_ctr = x_data_tag_all.groupby(["title", "tag", "prefix"], as_index=False)["label"].agg({"title_tag_prefix_ctr":"mean"})
    title_tag_prefix_count = x_data_tag_all.groupby(["title", "tag", "prefix"], as_index=False)["label"].agg({"title_tag_prefix_count":"count"})
    tag_len_prefix_len_title_ctr = x_data_tag_all.groupby(["tag","len_prefix", "len_title"], as_index=False)["label"].agg({"tag_len_prefix_len_title_ctr":"mean"})
    tag_len_prefix_len_title_count = x_data_tag_all.groupby(["tag","len_prefix", "len_title"], as_index=False)["label"].agg({"tag_len_prefix_len_title_count":"count"})
    tag_len_query_prediction_ctr = x_data_tag_all.groupby(['len_query_prediction', "tag"], as_index=False)["label"].agg({"tag_len_query_prediction_ctr":"mean"})
    train_tag_statistics = get_tag_statistics(x_data_tag_all, tag_ctr, tag_title_ctr, tag_title_count, tag_len_title_ctr, tag_len_title_count, tag_prefix_ctr, tag_len_prefix_ctr, tag_len_prefix_count, title_tag_prefix_ctr, title_tag_prefix_count, tag_len_prefix_len_title_ctr, tag_len_prefix_len_title_count, tag_len_query_prediction_ctr)
    x_data_tag_all = pd.concat([x_data_tag_all, train_tag_statistics], axis = 1)   
    print("附加文章标签统计值")
    tag_max_query_ctr = x_data_tag_all.groupby(["tag", "max_query"], as_index=False)["label"].agg({"tag_max_query_ctr":"mean"})
    tag_max_query_count = x_data_tag_all.groupby(["tag", "max_query"], as_index=False)["label"].agg({"tag_max_query_count":"count"})
    tag_len_max_query_ctr = x_data_tag_all.groupby(["tag", "len_max_query"], as_index=False)["label"].agg({"tag_len_max_query_ctr":"mean"})
    tag_len_max_query_count = x_data_tag_all.groupby(["tag", "len_max_query"], as_index=False)["label"].agg({"tag_len_max_query_count":"count"})
    title_tag_max_query_ctr = x_data_tag_all.groupby(["title", "tag", "max_query"], as_index=False)["label"].agg({"title_tag_max_query_ctr":"mean"})
    title_tag_max_query_count = x_data_tag_all.groupby(["title", "tag", "max_query"], as_index=False)["label"].agg({"title_tag_max_query_count":"count"})
    tag_len_max_query_len_title_ctr = x_data_tag_all.groupby(["tag","len_max_query", "len_title"], as_index=False)["label"].agg({"tag_len_max_query_len_title_ctr":"mean"})
    tag_len_max_query_len_title_count = x_data_tag_all.groupby(["tag","len_max_query", "len_title"], as_index=False)["label"].agg({"tag_len_max_query_len_title_count":"count"})
    train_max_query_tag_statistics = get_max_query_tag_statistics(x_data_tag_all, tag_max_query_ctr, tag_max_query_count, tag_len_max_query_ctr, tag_len_max_query_count, title_tag_max_query_ctr, title_tag_max_query_count, tag_len_max_query_len_title_ctr, tag_len_max_query_len_title_count)
    x_data_tag_all = pd.concat([x_data_tag_all, train_max_query_tag_statistics], axis = 1)
    train_list_sentences = get_list_sentences(x_data_tag_all)
    x_data_tag_all = pd.concat([x_data_tag_all, train_list_sentences], axis = 1)
    
    
    #print("存储训练集")
    #x_data_tag_all.to_csv("./x_data.csv", index = False)
    print("验证集")
    #vali_data = pd.read_table("./oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    vali_data = pd.read_table(vali, sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    x_vali_tag_all = get_query_prediction_statistics(vali_data.fillna('{}'))
    vali_statistics = get_statistics(x_vali_tag_all)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_statistics], axis = 1)   
    vali_max_query_statistics = get_max_query_statistics(x_vali_tag_all, max_query_ctr, max_query_count, len_max_query_ctr, len_max_query_count, len_query_prediction_max_query_ctr, len_query_prediction_max_query_count)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_max_query_statistics], axis = 1)
    vali_prefix_statistics = get_prefix_statistics(x_vali_tag_all, prefix_ctr, prefix_count, len_prefix_ctr, len_prefix_count, len_query_prediction_prefix_ctr, len_query_prediction_prefix_count)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_prefix_statistics], axis = 1)
    vali_title_statistics = get_title_statistics(x_vali_tag_all, title_ctr, title_count, len_title_ctr, len_title_count)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_title_statistics], axis = 1) 
    vali_tag_statistics = get_tag_statistics(x_vali_tag_all, tag_ctr, tag_title_ctr, tag_title_count, tag_len_title_ctr, tag_len_title_count, tag_prefix_ctr, tag_len_prefix_ctr, tag_len_prefix_count, title_tag_prefix_ctr, title_tag_prefix_count, tag_len_prefix_len_title_ctr, tag_len_prefix_len_title_count, tag_len_query_prediction_ctr)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_tag_statistics], axis = 1)   
    vali_max_query_tag_statistics = get_max_query_tag_statistics(x_vali_tag_all, tag_max_query_ctr, tag_max_query_count, tag_len_max_query_ctr, tag_len_max_query_count, title_tag_max_query_ctr, title_tag_max_query_count, tag_len_max_query_len_title_ctr, tag_len_max_query_len_title_count)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_max_query_tag_statistics], axis = 1)
    vali_list_sentences = get_list_sentences(x_vali_tag_all)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_list_sentences], axis = 1)
    
    #print("存储验证集")
    #x_vali_tag_all.to_csv("./x_vali.csv", index = False)
    print("测试集")
    #testA_data = pd.read_table("./oppo_round1_test_A_20180929/oppo_round1_test_A_20180929.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    #testB_data = pd.read_table("./oppo_round1_test_B_20181106.txt", sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    testB_data = pd.read_table(testB, sep = "\t", encoding = "utf-8", quoting =csv.QUOTE_NONE, header=None, names=["prefix", "query_prediction", "title", "tag", "label"], error_bad_lines=False)
    x_testB_tag_all = get_query_prediction_statistics(testB_data.fillna('{}'))
    test_statistics = get_statistics(x_testB_tag_all)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_statistics], axis = 1)   
    test_max_query_statistics = get_max_query_statistics(x_testB_tag_all, max_query_ctr, max_query_count, len_max_query_ctr, len_max_query_count, len_query_prediction_max_query_ctr, len_query_prediction_max_query_count)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_max_query_statistics], axis = 1)
    test_prefix_statistics = get_prefix_statistics(x_testB_tag_all, prefix_ctr, prefix_count, len_prefix_ctr, len_prefix_count, len_query_prediction_prefix_ctr, len_query_prediction_prefix_count)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_prefix_statistics], axis = 1)
    test_title_statistics = get_title_statistics(x_testB_tag_all, title_ctr, title_count, len_title_ctr, len_title_count)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_title_statistics], axis = 1) 
    test_tag_statistics = get_tag_statistics(x_testB_tag_all, tag_ctr, tag_title_ctr, tag_title_count, tag_len_title_ctr, tag_len_title_count, tag_prefix_ctr, tag_len_prefix_ctr, tag_len_prefix_count, title_tag_prefix_ctr, title_tag_prefix_count, tag_len_prefix_len_title_ctr, tag_len_prefix_len_title_count, tag_len_query_prediction_ctr)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_tag_statistics], axis = 1)      
    test_max_query_tag_statistics = get_max_query_tag_statistics(x_testB_tag_all, tag_max_query_ctr, tag_max_query_count, tag_len_max_query_ctr, tag_len_max_query_count, title_tag_max_query_ctr, title_tag_max_query_count, tag_len_max_query_len_title_ctr, tag_len_max_query_len_title_count)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_max_query_tag_statistics], axis = 1)
    test_list_sentences = get_list_sentences(x_testB_tag_all)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_list_sentences], axis = 1)   
    
    #svd
    train_prefix_vec, train_title_vec, train_max_query_vec, vali_prefix_vec, vali_title_vec, vali_max_query_vec, test_prefix_vec, test_title_vec, test_max_query_vec = get_feature.svd_model(x_data_tag_all, x_vali_tag_all, x_testB_tag_all)

    train_prefix_title_svd_data = get_feature.get_prefix_title_svd_data(train_prefix_vec, train_title_vec)
    train_prefix_max_query_svd_data = get_feature.get_prefix_max_query_svd_data(train_prefix_vec, train_max_query_vec)
    train_max_query_title_svd_data = get_feature.get_max_query_title_svd_data(train_max_query_vec, train_title_vec)
    x_data_tag_all = pd.concat([x_data_tag_all, train_prefix_title_svd_data, train_prefix_max_query_svd_data, train_max_query_title_svd_data], axis = 1)

    vali_prefix_title_svd_data = get_feature.get_prefix_title_svd_data(vali_prefix_vec, vali_title_vec)
    vali_prefix_max_query_svd_data = get_feature.get_prefix_max_query_svd_data(vali_prefix_vec, vali_max_query_vec)
    vali_max_query_title_svd_data = get_feature.get_max_query_title_svd_data(vali_max_query_vec, vali_title_vec)
    x_vali_tag_all = pd.concat([x_vali_tag_all, vali_prefix_title_svd_data, vali_prefix_max_query_svd_data, vali_max_query_title_svd_data], axis = 1)
    
    test_prefix_title_svd_data = get_feature.get_prefix_title_svd_data(test_prefix_vec, test_title_vec)
    test_prefix_max_query_svd_data = get_feature.get_prefix_max_query_svd_data(test_prefix_vec, test_max_query_vec)
    test_max_query_title_svd_data = get_feature.get_max_query_title_svd_data(test_max_query_vec, test_title_vec)
    x_testB_tag_all = pd.concat([x_testB_tag_all, test_prefix_title_svd_data, test_prefix_max_query_svd_data, test_max_query_title_svd_data], axis = 1)
    
    #随机森林
    model.RandomForestModel(x_data_tag_all, x_vali_tag_all, x_testB_tag_all)




