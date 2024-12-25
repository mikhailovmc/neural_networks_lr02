import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from navec import Navec
import joblib

from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.report.report_deco import ReportDecoNLP
from lightautoml.tasks import Task

N_THREADS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 9000
TARGET_NAME = 'sentiment'

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


def f1_macro(y_true, y_pred):
    return f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')


task = Task('multiclass', metric=f1_macro)

roles = {'target': TARGET_NAME, 'text': ['review']}


def net_train(file_name, net_name):
    data = pd.read_csv(file_name, sep='\t')
    data.head()

    data.sentiment.value_counts()

    data['review'].str.split(' ').apply(len).hist(bins=100)
    plt.show()

    train_data, test_data = train_test_split(data,
                                             test_size=TEST_SIZE,
                                             stratify=data[TARGET_NAME],
                                             random_state=RANDOM_STATE)

    train_data = train_data.sample(n=25_000, random_state=RANDOM_STATE)
    print('Data splitted. Parts sizes: train_data = {}, test_data = {}'.format(train_data.shape, test_data.shape))

    train, valid = train_test_split(train_data,
                                    test_size=TEST_SIZE,
                                    stratify=train_data[TARGET_NAME],
                                    random_state=RANDOM_STATE)

    print('Data splitted. Parts sizes: train = {}, valid = {}'.format(train.shape, valid.shape))

    path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(path)

    RD = ReportDecoNLP(output_path='report2')

    start = time.time()
    automl = TabularNLPAutoML(task=task,
                              timeout=TIMEOUT,
                              cpu_limit=N_THREADS,
                              general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb']]},
                              linear_pipeline_params={'text_features': "tfidf"},
                              gbm_pipeline_params={'text_features': 'embed'},
                              text_params={'lang': 'ru', 'bert_model': 'DeepPavlov/rubert-base-cased-conversational'},
                              autonlp_params={'model_name': 'random_lstm',
                                              'embedding_model': navec,
                                              'transformer_params': {'dataset_params': {
                                                  'max_length': 150,
                                                  'embed_size': 300},
                                              }
                                              },
                              tfidf_params={'svd': True, 'tfidf_params': {'ngram_range': (1, 1)}}

                              )

    automl_rd = RD(automl)

    oof_pred = automl_rd.fit_predict(train, valid_data=valid, roles=roles)
    print('oof_pred:\n{}\nShape = {}'.format(oof_pred, oof_pred.shape))
    time_automl = time.time() - start

    test_pred = automl_rd.predict(test_data)
    print('Prediction for test data:\n{}\nShape = {}'.format(test_pred, test_pred.shape))

    print('Check scores...')
    print('VALID score: {}'.format(f1_macro(valid[TARGET_NAME].map(automl_rd.reader.class_mapping).values, oof_pred.data)))
    test_automl = f1_macro(test_data[TARGET_NAME].map(automl_rd.reader.class_mapping).values, test_pred.data)
    print('TEST score: {}'.format(test_automl))
    print('TIME: {}'.format(time_automl))

    print('recall: {}'.format(recall_score(test_data[TARGET_NAME].map(automl_rd.reader.class_mapping).values,
                                           np.argmax(test_pred.data, axis=1), average="macro")))
    print('precision: {}'.format(precision_score(test_data[TARGET_NAME].map(automl_rd.reader.class_mapping).values,
                                                 np.argmax(test_pred.data, axis=1), average="macro")))
    joblib.dump(automl_rd, net_name)
