import spacy
import torch

# import keras
import pandas
import matplotlib
import numpy

import platform

from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import json


def tf_calculator(tf_doc):
    counts = {}
    for word in tf_doc:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] = counts[word] + 1

    frequency = {}

    for word in tf_doc:
        if word not in frequency:
            frequency[word] = counts[word]/len(tf_doc)
        else:
            continue

    return frequency

client = MongoClient("")

db = client.ApplyAI

job_data_raw = db.JobDataRaw

job_data_clean = db.JobDataClean

all_clean_data = job_data_clean.find({}, {'_id': 0, 'Company': 0})
for data in all_clean_data:
    print(data['Description'])

corpus_idf = db.CorpusIDF

_idf = corpus_idf.find_one({}, {'_id': 0})

# corpus_idf.delete_one(_idf)

# print(corpus_idf.find_one({}, {'_id': 0}))

with open("/Users/bkirkham/FinalProject/jobs_data.json", 'r', encoding="utf-8") as f:
    data_raw = json.load(f)

with open("/Users/bkirkham/FinalProject/clean_jobs_data.json", 'r', encoding="utf-8") as f:
    data_clean = json.load(f)

# print(data_raw[0])
# print(data_clean[0])

corpus = []

for data in data_clean:
    document = data['Description']
    # for word in document:
    #     if word not in corpus:
    #         corpus.append(word)
    document = "".join(word + " " for word in document)
    # print(document + '\n')
    corpus.append(document)

# print(corpus)

vectorizor = TfidfVectorizer(max_df=.7, min_df=2, smooth_idf=True, tokenizer=lambda x: x.split())
result = vectorizor.fit_transform(corpus)

# print(result)
feature = {}
for ele1, ele2 in zip(vectorizor.get_feature_names_out(), vectorizor.idf_):
    feature[ele1] = ele2
    # print(ele1, ':', ele2)

# corpus_idf.insert_one(feature)
# print(feature)

test_tf = tf_calculator(data_clean[0]['Description'])

# print(test_tf)

test_tf_idf = {}

for word in data_clean[0]['Description']:
    if word in feature:
        tf_idf = test_tf[word] * feature[word]
        test_tf_idf[word] = tf_idf
    else:
        test_tf_idf[word] = test_tf[word]

sorted_tfidf = sorted(test_tf_idf.items(), key=lambda x: x[1], reverse=False)
# print(len(feature))
# print(len(test_tf))
# print(len(test_tf_idf))
# print(len(data_clean[0]['Description']))


#i need to reload the job data i have
# for data in data_raw:
#     job_data_raw.insert_one(data)
#
# for data in data_clean:
#     job_data_clean.insert_one(data)











# old attempt at bag of words

# df = pandas.read_json("/Users/bkirkham/FinalProject/clean_jobs_data.json", orient='records')
#
# lists_of_words = df['Description']
# bag_of_words = {}
#
# for list in lists_of_words:
#     for word in list:
#         if word not in bag_of_words:
#             bag_of_words[word] = 1
#         else:
#             bag_of_words[word] = bag_of_words[word] + 1
#
# # print(lists_of_words.head(5))
# sorted_bag = sorted(bag_of_words.items(), key=lambda item: item[1])
# sorted_bag.reverse()
#
# print(sorted_bag)
# print(len(bag_of_words))
#
# print(platform.architecture())
