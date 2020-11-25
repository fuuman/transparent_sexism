import re
import os
import shap # use downloaded package instead of local package
import spacy
from joblib import Parallel, delayed
import utils
# import torch
import time
from sklearn.model_selection import train_test_split
# from fastbert import FastBERT
import pandas as pd
from _utils.fastbert import get_fastbert_model
import pickle
# import lstm as lc
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from tqdm import tqdm
import numpy as np
from functools import partial
from _data.unsex_data import UnsexData
from collections import OrderedDict
from _utils.pathfinder import get_repo_path
from sklearn.metrics import accuracy_score
from _utils.pathfinder import get_repo_path, get_experiment_path
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

### save_dir
REPO_DIR = get_repo_path()
DATA_ROOT = os.path.join(REPO_DIR, '_explanations')

SAVE_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
SAVE_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
SAVE_SST_DIR = os.path.join(DATA_ROOT, 'sst')
SAVE_UNSEX_DIR = os.path.join(DATA_ROOT, 'unsex')


def get_unsex_svm_coef_d(pipeline):
    classifier = pipeline.named_steps['classifier']
    feature = pipeline.named_steps['tfidf']
    coefficients = classifier.coef_[0]
    vocabulary = feature.vocabulary_
    d = {}
    for word, index in vocabulary.items():
        score = float(coefficients[index])
        d[str(word)] = score
    return d

def save_unsex_svm_coef(model_name, experiment=None):
    if experiment:
        path = os.path.join(get_experiment_path(experiment), 'models', '{}.pkl'.format(model_name))
    else:
        model_path = '_trained_models/{}.pkl'.format(model_name)
        path = utils.get_abs_path(REPO_DIR, model_path)
    print('model path: {}'.format(path))
    pipeline = utils.load_pickle(path, encoding=False)
    svm_coef_d = get_unsex_svm_coef_d(pipeline)
    if experiment:
        features = 'explanations/features/{}_builtin_all_features.pkl'.format(model_name)
        path = os.path.join(get_experiment_path(experiment), features)
    else:
        features = 'features/{}_builtin_all_features.pkl'.format(model_name)
        path = utils.get_abs_path(SAVE_UNSEX_DIR, features)
    utils.save_pickle(svm_coef_d, path)

def get_unsex_xgb_impt_d(pipeline):
    importance = pipeline.named_steps['classifier'].feature_importances_
    vocab = pipeline.named_steps['tfidf'].vocabulary_
    d = {}
    for word, index in vocab.items():
        score = float(importance[index])
        d[str(word)] = score
    return d

def save_unsex_xgb_impt(model_name, experiment=None):
    if experiment:
        path = os.path.join(get_experiment_path(experiment), 'models', '{}.pkl'.format(model_name))
    else:
        model = '_trained_models/{}.pkl'.format(model_name)
        path = utils.get_abs_path(REPO_DIR, model)
    print('model path: {}'.format(path))
    pipeline = utils.load_pickle(path)
    xgb_impt_d = get_unsex_xgb_impt_d(pipeline)
    if experiment:
        features = 'explanations/features/{}_builtin_all_features.pkl'.format(model_name)
        path = os.path.join(get_experiment_path(experiment), features)
    else:
        features = 'features/{}_builtin_all_features.pkl'.format(model_name)
        path = utils.get_abs_path(SAVE_UNSEX_DIR, features)
    utils.save_pickle(xgb_impt_d, path)


def save_unsex_lr_impt(model_name, experiment=None):
    if experiment:
        path = os.path.join(get_experiment_path(experiment), 'models', '{}.pkl'.format(model_name))
    else:
        model = '_trained_models/{}.pkl'.format(model_name)
        path = utils.get_abs_path(REPO_DIR, model)
    print('model path: {}'.format(path))
    pipeline = utils.load_pickle(path)
    lr_impt_d = get_unsex_lr_impt_d(pipeline)
    if experiment:
        features = 'explanations/features/{}_builtin_all_features.pkl'.format(model_name)
        path = os.path.join(get_experiment_path(experiment), features)
    else:
        features = 'features/{}_builtin_all_features.pkl'.format(model_name)
        path = utils.get_abs_path(SAVE_UNSEX_DIR, features)
    utils.save_pickle(lr_impt_d, path)


def get_unsex_lr_impt_d(pipeline):
    importance = pipeline.named_steps['classifier'].coef_[0]
    vocab = pipeline.named_steps['tfidf'].vocabulary_
    d = {}
    for word, index in vocab.items():
        score = float(importance[index])
        d[str(word)] = score
    return d




def wrapper_clf_predict(test_tokens, model=None, model_name=None):
    labels = []
    if model_name == 'lstm_att':
        test_split_tokens = split_tokens(test_tokens)
        mapping = [model.get_words_to_ids(l) for l in test_split_tokens]
        labels = np.array(model.predict(test_split_tokens, mapping, return_probablity=True))
    elif model_name == "xgb":
        labels = model.predict_proba(test_tokens)
    else:
        labels = model.predict(test_tokens)
        labels = np.array([[0.999, 0.001] if l == -1 else [0.001, 0.999] for l in labels])
    return labels

def unsex_wrapper_clf_predict(test_tokens, model=None, model_name=None):
    labels = []
    if model_name == 'lstm_att':
        test_split_tokens = split_tokens(test_tokens)
        mapping = [model.get_words_to_ids(l) for l in test_split_tokens]
        labels = np.array(model.predict(test_split_tokens, mapping, return_probablity=True))
    elif model_name == "xgboost":
        labels = model.predict_proba(test_tokens)
    elif 'bert' in model_name:
        labels = model.predict_batch(test_tokens)
        labels = [sorted(lp, key=lambda i: i[0]) for lp in labels]
        labels = np.array([[j[0][1], j[1][1]] for j in labels])
    else:
        labels = model.predict(test_tokens)
        labels = np.array([[0.999, 0.001] if l == -1 else [0.001, 0.999] for l in labels])
    return labels

def get_lime(model, test_tokens, model_name):
    explainer = LimeTextExplainer(class_names=["genuine", "deceptive"],
                                  split_expression=u'\s+')
    W = []
    for idx, text in enumerate(test_tokens):
        print(idx, text)
        tmp_d = {}
        for i in text.split():
            tmp_d[i] = 1
        print(' now explain')
        exp = explainer.explain_instance(text, 
                                         partial(wrapper_clf_predict, model=model, model_name=model_name), 
                                         num_features=len(text.split()), 
                                         num_samples=1000)
        if len(tmp_d) != len(exp.as_list()):
            print(idx, len(tmp_d), len(dict(exp.as_list())))
        W.append(dict(exp.as_list()))
        if (idx+1) % 10 == 0:
            print('{} instances have been processed..'.format(idx+1))
    features_l, scores_l = [], []
    for d in W:
        features, scores = [], []
        for key, score in d.items():
            features.append(key)
            tmp = ' '.join(features)
            scores.append(score) # abs value should be taken subsequently
        features_l.append(tmp)
        scores_l.append(scores)
    return features_l, scores_l

def run_explain(text, idx, model, model_name):
    explainer = LimeTextExplainer(class_names=["non-sexist", "sexist"], split_expression=u'\W+')
    if len(text) == 0:
        return {'': 0.0}
    tmp_d = {}
    for i in text.split():
        tmp_d[i] = 1
    exp = explainer.explain_instance(text,
                                     partial(unsex_wrapper_clf_predict, model=model, model_name=model_name),
                                     num_features=len(text.split()),
                                     num_samples=1000)
    # print(f'explain finished for #{idx + 1}/816: {text}')
    if len(tmp_d) != len(exp.as_list()):
        print(idx, len(tmp_d), len(dict(exp.as_list())))
    return dict(exp.as_list())


def get_unsex_lime(model, test_tokens, model_name):
    explainer = LimeTextExplainer(class_names=["non-sexist", "sexist"])
    W = []
    for idx, text in tqdm(enumerate(test_tokens)):
        if len(text) == 0:
            W.append({'': 0.0})
            continue
        tmp_d = {}
        for i in text.split():
            tmp_d[i] = 1
        exp = explainer.explain_instance(text,
                                         partial(unsex_wrapper_clf_predict, model=model, model_name=model_name),
                                         num_features=len(text.split()),
                                         num_samples=1000)
        if len(tmp_d) != len(exp.as_list()):
            print(idx, len(tmp_d), len(dict(exp.as_list())))
        W.append(dict(exp.as_list()))
        # if (idx+1) % 10 == 0:
        #     print('{} instances have been processed..'.format(idx+1))
    # W = Parallel(n_jobs=4)(delayed(run_explain)(text, idx, model, model_name) for idx, text in enumerate(test_tokens[:10]))
    # W = [run_explain(text, idx, model, model_name) for idx, text in enumerate(test_tokens)]

    features_l, scores_l = [], []
    for d in W:
        features, scores = [], []
        for key, score in d.items():
            features.append(key)
            tmp = ' '.join(features)
            scores.append(score) # abs value should be taken subsequently
        features_l.append(tmp)
        scores_l.append(scores)
    return features_l, scores_l

def save_lime_coef(filename, model_name, SAVE_DIR, train_dev_tokens, test_tokens, d_file=None):
    model = 'models/{}.pkl'.format(filename)
    path = utils.get_abs_path(SAVE_DIR, model)
    if 'svm' in model_name:
        model = utils.load_pickle(path, encoding=False)
    else:
        if model_name == 'lstm_att':
            hp_d = 'models/{}.pkl'.format(d_file)
            hp_path = utils.get_abs_path(SAVE_DIR, hp_d)
            d = utils.load_pickle(hp_path)
            model = init_model(train_dev_tokens, d, path)
        else:
            model = utils.load_pickle(path)
    features_l, importance_l = get_lime(model, test_tokens, model_name)
    features = 'features/{}_lime_all_features.pkl'.format(model_name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(features_l, path)
    scores = 'feature_importance/{}_lime_all_scores.pkl'.format(model_name)
    path = utils.get_abs_path(SAVE_DIR, scores)
    utils.save_pickle(importance_l, path)


def save_unsex_lime_coef(model_name, test_tokens, experiment=None):
    if 'bert' in model_name:
        model = get_fastbert_model()
    else:
        if experiment:
            path = os.path.join(get_experiment_path(experiment), 'models', '{}.pkl'.format(model_name))
        else:
            model = '_trained_models/{}.pkl'.format(model_name)
            path = utils.get_abs_path(REPO_DIR, model)
        if 'svm' in model_name:
            model = utils.load_pickle(path, encoding=False)
        else:
            model = utils.load_pickle(path)
    features_l, importance_l = get_unsex_lime(model, test_tokens, model_name)
    if experiment:
        path = os.path.join(get_experiment_path(experiment), 'explanations/features/{}_lime_all_features.pkl'.format(model_name))
        utils.save_pickle(features_l, os.path.join(get_repo_path(), path))
        path = os.path.join(get_experiment_path(experiment), 'explanations/feature_importance/{}_lime_all_scores.pkl'.format(model_name))
        utils.save_pickle(importance_l, os.path.join(get_repo_path(), path))
    else:
        path = '_explanations/unsex/features/{}_lime_all_features.pkl'.format(model_name)
        utils.save_pickle(features_l, os.path.join(get_repo_path(), path))
        path = '_explanations/unsex/feature_importance/{}_lime_all_scores.pkl'.format(model_name)
        utils.save_pickle(importance_l, os.path.join(get_repo_path(), path))


def split_tokens(l):
    return [i.split() for i in l]

def init_model(train_dev_tokens, d, path):
    tokens = split_tokens(train_dev_tokens)
    model = lc.LSTMAttentionClassifier(tokens, 
                                       emb_dim=d['emb_dim'],
                                       hidden_dim=d['hidden_dim'],
                                       num_layers=d['num_layers'],
                                       min_count=d['min_count'],
                                       bidirectional=True)
    model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

def get_shap(clf_name, pipeline, train_dev_tokens, test_tokens):
    feature = pipeline.named_steps['feature']
    clf = pipeline.named_steps['clf']
    vocab = feature.vocabulary_
    index_feature_d = {}
    for word, index in vocab.items():
        index_feature_d[index] = word
    X_train = feature.transform(train_dev_tokens)
    X_test = feature.transform(test_tokens).toarray()
    explainer = None
    if 'svm' in clf_name:
        explainer = shap.LinearExplainer(clf, X_train, feature_dependence="independent")
    else:
        explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # get all features
    features_l, importance_l = [], []
    for idx, row in enumerate(shap_values):
        word_shap_val_d = {}
        for idx_b, shap_val in enumerate(row):
            feature = index_feature_d[idx_b]
            word_shap_val_d[feature] = abs(shap_val) # taking absolute value
        features_tmp = list(word_shap_val_d.keys())
        features = " ".join(features_tmp)
        features_l.append(features)
        scores = list(word_shap_val_d.values())
        importance_l.append(scores)
    return features_l, importance_l

def save_shap_val(file, name, SAVE_DIR, train_data, test_data):
    model = 'models/{}.pkl'.format(file)
    path = utils.get_abs_path(SAVE_DIR, model)
    print('model path: {}'.format(path))
    model = None
    if file == 'svm':
        model = utils.load_pickle(path, encoding=False)
    else:
        model = utils.load_pickle(path)
    features_l, importance_l = [], []
    if 'svm' in name:
        features_l, importance_l = get_shap('svm', model, train_dev_tokens, test_tokens)
    elif 'xgb' in name:
        features_l, importance_l = get_shap('xgb', model, train_dev_tokens, test_tokens)
    features = 'features/{}_shap_all_features.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(features_l, path)
    scores = 'feature_importance/{}_shap_all_scores.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, scores)
    utils.save_pickle(importance_l, path)


def get_unsex_shap(clf_name, pipeline, train_tokens, test_tokens):
    if 'bert' in clf_name:
        feature = TfidfVectorizer(stop_words='english')
        feature.fit(train_tokens)
        # X_test = pd.DataFrame(test_tokens)
        predict_wrapper = lambda ts: np.array([int(p[0][0]) for p in pipeline.predict_batch([x[0] for x in ts])])
    else:
        feature = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['classifier']
    X_test = feature.transform(test_tokens)
    X_train = feature.transform(train_tokens)
    vocab = feature.vocabulary_
    index_feature_d = {}
    for word, index in vocab.items():
        index_feature_d[index] = word

    if 'svm' in clf_name:
        # explainer = shap.LinearExplainer(clf, data=X_train, feature_dependence="independent")
        explainer = shap.LinearExplainer(clf, data=np.zeros(X_train.shape), feature_dependence="independent")
    elif 'lr' in clf_name:
        # explainer = shap.LinearExplainer(clf, data=X_train, feature_dependence="independent")
        explainer = shap.LinearExplainer(clf, data=np.zeros(X_train.shape), feature_dependence="independent")
    elif 'bert' in clf_name:
        explainer = shap.KernelExplainer(predict_wrapper, np.array([[0]]), feature_dependence='independent')
    else:
        # explainer = shap.TreeExplainer(clf, data=None, feature_dependence='independent')
        explainer = shap.TreeExplainer(clf, data=None, feature_dependence='independent')

    shap_values = explainer.shap_values(X_test)
    # get all features

    features_l, importance_l = [], []
    for idx, row in enumerate(shap_values):
        word_shap_val_d = {}
        for idx_b, shap_val in enumerate(row):
            feature = index_feature_d[idx_b]
            word_shap_val_d[feature] = abs(shap_val)  # taking absolute value
        features_tmp = list(word_shap_val_d.keys())
        features = " ".join(features_tmp)
        features_l.append(features)
        scores = list(word_shap_val_d.values())
        importance_l.append(scores)
    return features_l, importance_l


def save_unsex_shap_val(name, train_tokens, test_tokens, experiment=None):
    if 'bert' in name:
        model = get_fastbert_model()
    else:
        if experiment:
            path = os.path.join(get_experiment_path(experiment), 'models', '{}.pkl'.format(name))
        else:
            model = '_trained_models/{}.pkl'.format(name)
            path = utils.get_abs_path(REPO_DIR, model)
        print('model path: {}'.format(path))
        if 'svm' in name:
            model = utils.load_pickle(path, encoding=False)
        else:
            model = utils.load_pickle(path)
    features_l, importance_l = get_unsex_shap(name, model, train_tokens, test_tokens)
    if experiment:
        path = os.path.join(get_experiment_path(experiment), 'explanations', 'features/{}_shap_all_features.pkl'.format(name))
        utils.save_pickle(features_l, path)
        path = os.path.join(get_experiment_path(experiment), 'explanations', 'feature_importance/{}_shap_all_scores.pkl'.format(name))
        utils.save_pickle(importance_l, path)
    else:
        features = 'features/{}_shap_all_features.pkl'.format(name)
        path = utils.get_abs_path(SAVE_UNSEX_DIR, features)
        utils.save_pickle(features_l, path)
        scores = 'feature_importance/{}_shap_all_scores.pkl'.format(name)
        path = utils.get_abs_path(SAVE_UNSEX_DIR, scores)
        utils.save_pickle(importance_l, path)


def save_tokens(test_tokens):
    path = utils.get_abs_path(SAVE_UNSEX_DIR, 'tweets/tweets.pkl')
    utils.save_pickle(test_tokens, path)


def save_data(SAVE_DIR, train_dev_tokens, test_tokens):
    # save_svm_coef('svm', 'svm', SAVE_DIR)
    # save_svm_coef('svm_l1', 'svm_l1', SAVE_DIR)

    # save_xgb_impt('xgb', 'xgb', SAVE_DIR)

    # save_lime_coef('svm', 'svm', SAVE_DIR, train_dev_tokens, test_tokens)
    # save_lime_coef('svm_l1', 'svm_l1', SAVE_DIR, train_dev_tokens, test_tokens)
    # save_lime_coef('xgb', 'xgb', SAVE_DIR, train_dev_tokens, test_tokens)
    # save_lime_coef('lstm_att', 'lstm_att', SAVE_DIR, \
    #                train_dev_tokens, test_tokens, d_file='lstm_att_hp')

    save_shap_val('svm', 'svm', SAVE_DIR, train_dev_tokens, test_tokens)
    # save_shap_val('svm_l1', 'svm_l1', SAVE_DIR, train_dev_tokens, test_tokens)
    # save_shap_val('xgb', 'xgb', SAVE_DIR, train_dev_tokens, test_tokens)


def save_unsex_data(train_tokens, test_tokens, experiment=None):
    ### unsex me stuff
    # built-in
    # save_unsex_svm_coef('svm_l1')
    save_unsex_svm_coef('svm', experiment=experiment)
    save_unsex_lr_impt('lr', experiment=experiment)
    save_unsex_xgb_impt('xgboost', experiment=experiment)
    #
    # lime
    # save_unsex_lime_coef('svm_l1', test_tokens)
    save_unsex_lime_coef('svm', test_tokens, experiment=experiment)
    save_unsex_lime_coef('lr', test_tokens, experiment=experiment)
    save_unsex_lime_coef('xgboost', test_tokens, experiment=experiment)
    # save_unsex_lime_coef('fast-bert', test_tokens)

    # shap
    # training tokens just needed to explain fastbert with shap
    # save_unsex_shap_val('svm_l1', None, test_tokens)
    save_unsex_shap_val('svm', train_tokens, test_tokens, experiment=experiment)
    save_unsex_shap_val('lr', train_tokens, test_tokens, experiment=experiment)
    save_unsex_shap_val('xgboost', train_tokens, test_tokens, experiment=experiment)
    # save_unsex_shap_val('fast-bert', train_tokens, test_tokens)


def explain_all(X_train, X_test, experiment=None):
    # my stuff
    print('=== unsex binary ===')
    save_unsex_data(X_train, X_test, experiment=experiment)


if __name__ == '__main__':
    with open('_explanations/unsex/used_training_data/X_test_explainable.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('_explanations/unsex/used_training_data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    s = time.time()
    explain_all(X_train, X_test)
    print(time.time() - s)
