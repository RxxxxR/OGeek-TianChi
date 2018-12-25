# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:58:21 2018

@author: 20277
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

def get_max_query_prediction_key(query_prediction_json):
    if query_prediction_json!={}:
        max_query = max(query_prediction_json, key=query_prediction_json.get)
    else:
        max_query = "" 
    return max_query
    

def get_statistics_query_prediction(query_prediction_json, title):
    if query_prediction_json!={}:
        np_query_prediction_json = np.array(list(query_prediction_json.values()), dtype = np.float32)
        max_query_prediction_json = np.max(np_query_prediction_json)
        min_query_prediction_json = np.min(np_query_prediction_json)
        sum_query_prediction_json = np.sum(np_query_prediction_json)
        len_query_prediction_json = np_query_prediction_json.shape[0]
        med_query_prediction_json = np.median(np_query_prediction_json)
        var_query_prediction_json = np.var(np_query_prediction_json)
        mean_query_prediction_json = sum_query_prediction_json/len_query_prediction_json
        if sum_query_prediction_json>1:
            sum_query_prediction_json = 1
        query_prediction_title_rate = float(query_prediction_json.get(title,min([1-sum_query_prediction_json, min_query_prediction_json])))
    else:
        max_query_prediction_json, min_query_prediction_json, sum_query_prediction_json, mean_query_prediction_json, query_prediction_title_rate, len_query_prediction_json, med_query_prediction_json, var_query_prediction_json = (0, 0, 0, 0, 0, 0, 0, 0)
    return str([max_query_prediction_json, min_query_prediction_json, sum_query_prediction_json, mean_query_prediction_json, query_prediction_title_rate, len_query_prediction_json, med_query_prediction_json, var_query_prediction_json])[1:-1]


def get_statistics_new_query_prediction(query_prediction_json, title):
    new_query_prediction_json = {}
    title = str(title)
    for keys, values in query_prediction_json.items():
        keys = str(keys)
        new_query_prediction_json[keys] = (1 - levenshtein(keys, title)/(len(keys) + len(title)))*float(values)
    
    if new_query_prediction_json!={}:
        np_new_query_prediction_json = np.array(list(new_query_prediction_json.values()), dtype = np.float32)
        max_new_query_prediction_json = np.max(np_new_query_prediction_json)
        min_new_query_prediction_json = np.min(np_new_query_prediction_json)
        sum_new_query_prediction_json = np.sum(np_new_query_prediction_json)
        len_new_query_prediction_json = np_new_query_prediction_json.shape[0]
        med_new_query_prediction_json = np.median(np_new_query_prediction_json)
        var_new_query_prediction_json = np.var(np_new_query_prediction_json)
        if sum_new_query_prediction_json>1:
            sum_new_query_prediction_json=1
        mean_new_query_prediction_json = sum_new_query_prediction_json/len_new_query_prediction_json
    else:
        max_new_query_prediction_json, min_new_query_prediction_json, sum_new_query_prediction_json, mean_new_query_prediction_json, med_new_query_prediction_json, var_new_query_prediction_json= (0, 0, 0, 0, 0, 0)
    
    return str([max_new_query_prediction_json, min_new_query_prediction_json, sum_new_query_prediction_json, mean_new_query_prediction_json, med_new_query_prediction_json, var_new_query_prediction_json])[1:-1]


#提取有用的字符串
def char_cleaner(char):
    char = str(char)
    pattern = re.compile("[^a-zA-Z\u4E00-\u9FA5]")#t提取中文英文和数字
    char = re.sub(pattern, "", char)
    char = char.lower()
    return char

#前缀词占文章标题的比例
def get_prefix_title_rate(prefix, title):
    str_1 = char_cleaner(prefix)
    str_2 = char_cleaner(title)
    if str_1 in str_2:
        try:
            prefix_title_rate = len(str_1)/len(str_2)
        except:
            prefix_title_rate = 0
    else:
        prefix_title_rate = 0
    return prefix_title_rate

#编辑距离
def levenshtein(first, second):
    first = str(first).lower()
    second = str(second).lower()
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
            # print distance_matrix
    return distance_matrix[first_length - 1][second_length - 1]    
    


def get_prefix_title_svd_data(prefix_vec, title_vec):
    data = pd.DataFrame([])
    dot_sim = []
    consin_sim = []
    for i in range(prefix_vec.shape[0]):
        dot_sim.append(np.dot(prefix_vec[i], title_vec[i].T))
        consin_sim.append(cosine_similarity([prefix_vec[i], title_vec[i]])[0][1])
    data['prefix_title_dot'] = dot_sim
    data['prefix_title_consin'] = consin_sim
    search_query_abs = np.abs(prefix_vec - title_vec)
    search_query_mul = prefix_vec * title_vec
    data['prefix_title_abs_mean'] = search_query_abs.mean(1)
    data['prefix_title_abs_sum'] = search_query_abs.sum(1)
    data['prefix_title_abs_std'] = search_query_abs.std(1)
    data['prefix_title_abs_max'] = search_query_abs.max(1)
    data['prefix_title_abs_min'] = search_query_abs.min(1)
    data['prefix_title_mul_mean'] = search_query_mul.mean(1)
    data['prefix_title_mul_sum'] = search_query_mul.sum(1)
    data['prefix_title_mul_std'] = search_query_mul.std(1)
    data['prefix_title_mul_max'] = search_query_mul.max(1)
    data['prefix_title_mul_min'] = search_query_mul.min(1)
    return data

def get_prefix_max_query_svd_data(prefix_vec, max_query_vec):
    data = pd.DataFrame([])
    dot_sim = []
    consin_sim = []
    for i in range(prefix_vec.shape[0]):
        dot_sim.append(np.dot(prefix_vec[i], max_query_vec[i].T))
        consin_sim.append(cosine_similarity([prefix_vec[i], max_query_vec[i]])[0][1])
    data['prefix_max_query_dot'] = dot_sim
    data['prefix_max_query_consin'] = consin_sim
    search_query_abs = np.abs(prefix_vec - max_query_vec)
    search_query_mul = prefix_vec * max_query_vec
    data['prefix_max_query_abs_mean'] = search_query_abs.mean(1)
    data['prefix_max_query_abs_sum'] = search_query_abs.sum(1)
    data['prefix_max_query_abs_std'] = search_query_abs.std(1)
    data['prefix_max_query_abs_max'] = search_query_abs.max(1)
    data['prefix_max_query_abs_min'] = search_query_abs.min(1)
    data['prefix_max_query_mul_mean'] = search_query_mul.mean(1)
    data['prefix_max_query_mul_sum'] = search_query_mul.sum(1)
    data['prefix_max_query_mul_std'] = search_query_mul.std(1)
    data['prefix_max_query_mul_max'] = search_query_mul.max(1)
    data['prefix_max_query_mul_min'] = search_query_mul.min(1)
    return data

def get_max_query_title_svd_data(max_query_vec, title_vec):
    data = pd.DataFrame([])
    dot_sim = []
    consin_sim = []
    for i in range(max_query_vec.shape[0]):
        dot_sim.append(np.dot(max_query_vec[i], title_vec[i].T))
        consin_sim.append(cosine_similarity([max_query_vec[i], title_vec[i]])[0][1])
    data['max_query_title_dot'] = dot_sim
    data['max_query_title_consin'] = consin_sim
    search_query_abs = np.abs(max_query_vec - title_vec)
    search_query_mul = max_query_vec * title_vec
    data['max_query_title_abs_mean'] = search_query_abs.mean(1)
    data['max_query_title_abs_sum'] = search_query_abs.sum(1)
    data['max_query_title_abs_std'] = search_query_abs.std(1)
    data['max_query_title_abs_max'] = search_query_abs.max(1)
    data['max_query_title_abs_min'] = search_query_abs.min(1)
    data['max_query_title_mul_mean'] = search_query_mul.mean(1)
    data['max_query_title_mul_sum'] = search_query_mul.sum(1)
    data['max_query_title_mul_std'] = search_query_mul.std(1)
    data['max_query_title_mul_max'] = search_query_mul.max(1)
    data['max_query_title_mul_min'] = search_query_mul.min(1)
    return data


def svd_model(x_data_tag_all, x_vali_tag_all, x_test_tag_all):
    train_prefix_vec = x_data_tag_all['prefix_list'].tolist()
    train_title_vec = x_data_tag_all['title_list'].tolist()
    train_max_query_vec = x_data_tag_all['max_query_list'].fillna("").tolist()
    vali_prefix_vec = x_vali_tag_all['prefix_list'].tolist()
    vali_title_vec = x_vali_tag_all['title_list'].tolist()
    vali_max_query_vec = x_vali_tag_all['max_query_list'].fillna("").tolist()
    test_prefix_vec = x_test_tag_all['prefix_list'].tolist()
    test_title_vec = x_test_tag_all['title_list'].tolist()
    test_max_query_vec = x_test_tag_all['max_query_list'].fillna("").tolist()
    
    corpus = train_prefix_vec + train_title_vec + train_max_query_vec + \
        vali_prefix_vec + vali_title_vec + vali_max_query_vec + \
        test_prefix_vec + test_title_vec + test_max_query_vec
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
    normalizer = Normalizer(copy=False)
    svd = TruncatedSVD(30)
    lsa = make_pipeline(tfidf, normalizer, svd)
    lsa.fit(corpus)

    train_prefix_vec = lsa.transform(train_prefix_vec)
    train_title_vec = lsa.transform(train_title_vec)
    train_max_query_vec = lsa.transform(train_max_query_vec)
    
    vali_prefix_vec = lsa.transform(vali_prefix_vec)
    vali_title_vec = lsa.transform(vali_title_vec)
    vali_max_query_vec = lsa.transform(vali_max_query_vec)
    
    test_prefix_vec = lsa.transform(test_prefix_vec)
    test_title_vec = lsa.transform(test_title_vec)
    test_max_query_vec = lsa.transform(test_max_query_vec)
    
    return train_prefix_vec, train_title_vec, train_max_query_vec, vali_prefix_vec, vali_title_vec, vali_max_query_vec, test_prefix_vec, test_title_vec, test_max_query_vec 


