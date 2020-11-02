import os
import pickle
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from _utils.pathfinder import get_repo_path
from _data.unsex_data import UnsexData
from _utils.fastbert import get_fastbert_model


def _load_tweets():
    path = os.path.join(get_repo_path(), '_explanations', 'unsex')
    with open(os.path.join(path, 'used_training_data', 'X_test_raw.pkl'), 'rb') as f:
        tweets = pickle.load(f)
    return tweets


def _load_features_and_scores(model, ex_method):
    path = os.path.join(get_repo_path(), '_explanations', 'unsex')
    if ex_method in ['impt', 'coef']:
        with open(os.path.join(path, 'features', f'{model}_{ex_method}_all_features.pkl'), 'rb') as f:
            c = pickle.load(f)
            features = c.keys()
            scores = list(c.values())
        return features, scores
    else:
        with open(os.path.join(path, 'features', f'{model}_{ex_method}_all_features.pkl'), 'rb') as f:
            features = pickle.load(f)
        with open(os.path.join(path, 'feature_importance', f'{model}_{ex_method}_all_scores.pkl'), 'rb') as f:
            scores = pickle.load(f)
        return list(map(lambda i: i.split(), features)), scores


def _get_top_k_features(features, scores, k):
    comb = list(zip(features, scores))
    return [fe_sc[0] for fe_sc in sorted(comb, key=lambda i: i[1], reverse=True)][:k]


def _load_att_weights():
    path = os.path.join(get_repo_path(), '_explanations', 'unsex', 'attention_weights', 'fast-bert-att-all.npy')
    return np.load(path)


def print_prediction_and_label(tweet_id, model_name):
    classes = ['non-sexist', 'sexist']
    path = os.path.join(get_repo_path(), '_trained_models', f'{model_name}.pkl')
    if 'svm' in model_name:
        pipeline = utils.load_pickle(path, encoding=False)
    elif 'bert' in model_name:
        pipeline = get_fastbert_model()
    else:
        pipeline = utils.load_pickle(path)
    tweet = _load_tweets()[tweet_id]
    ud = UnsexData()
    X_train, X_test, y_train, y_test = ud.get_preprocessed_data()
    # label_idx = labels[tweets.index(tweet)]
    if 'bert' in model_name:
        prediction_idx = int(pipeline.predict_batch([tweet])[0][0][0])
    else:
        prediction_idx = pipeline.predict([tweet])[0]
    # print(f'###### Classes ######')
    print(f'Prediction of {model_name.upper()}: {classes[prediction_idx]}')
    # print(f'Real class: {classes[label_idx]}')


def create_tfidf_vectorizer():
    tfidf = TfidfVectorizer()
    with open(os.path.join(get_repo_path(), '_explanations', 'unsex', 'used_training_data', 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    tfidf.fit(X_train)
    return tfidf


def generate_explanation(tweet_id, tweet, f, s, ex_method, k=3):
    if ex_method in ['coef', 'impt']:
        top_global_features = _get_top_k_features(f, s, 30)
        exp = [f for f in top_global_features if f in tweet.lower()]
        if not exp:
            exp = ['']
    else:
        fs = [(f, s) for f, s in zip(f[tweet_id], s[tweet_id]) if f in tweet.split()]
        f = [i[0] for i in fs]
        s = [i[1] for i in fs]
        exp = _get_top_k_features(f, s, k)
    return exp


def plot_cs(matrix):
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()


tfidf = create_tfidf_vectorizer()

if __name__ == '__main__':
    # explainable chart
    # p1 = plt.bar(['LR', 'XGBoost', 'SVM', 'FastBERT'], [522, 662, 523, 577])
    # p2 = plt.bar(['LR', 'XGBoost', 'SVM', 'FastBERT'], [816-522, 816-662, 816-523, 816-577], bottom=[522, 662, 523, 577])
    # plt.legend((p1[0], p2[0]), ('Explainable', 'Unexplainable'))
    # plt.show()

    # current options: svm, svm_l1, lr, xgboost
    model_name = 'xgboost'

    # filter "unimportant tweets" without non-zero LIME explanations
    # tweets with lime explanation that makes sense (!= [0,0,0,0,...])
    s = _load_features_and_scores(model_name, 'lime')[1]
    tweet_ids = [tweet_id for tweet_id, scores in enumerate(s) if sum(scores) != 0]
    # print(len(tweet_ids), '/', len(s))
    tweets = _load_tweets()
    explainable_tweets = [(i, t) for i, t in enumerate(tweets) if i in tweet_ids]

########################
    exps = {}

    if 'svm' in model_name:
        ex_methods = ['lime', 'shap', 'coef']
    elif model_name in ['xgboost', 'lr']:
        ex_methods = ['lime', 'shap', 'impt']
    else:
        # fastbert todo
        ex_methods = []

    for ex_method in ex_methods:
        exps[ex_method] = []
        f, s = _load_features_and_scores(model_name, ex_method)
        for i, t in tqdm(explainable_tweets):
            exps[ex_method].append(generate_explanation(i, t, f, s, ex_method))
        exps[ex_method] = np.array(list(map(lambda x: ' '.join(x), exps[ex_method])))
        exps[ex_method] = tfidf.transform(exps[ex_method])

    # per ex_method #######
    cs_values_lime_shap = []
    cs_values_lime_impt = []
    cs_values_shap_impt = []
    for i in range(len(explainable_tweets)):
        lime_array = exps['lime'].toarray()[i:i+1, :]
        shap_array = exps['shap'].toarray()[i:i+1, :]
        impt_array = exps['impt'].toarray()[i:i+1, :]
        matrix = np.concatenate((lime_array, shap_array, impt_array))
        cs_m = cosine_similarity(matrix)
        cs_values_lime_shap.append(cs_m[0:1, 1:2][0][0])
        cs_values_lime_impt.append(cs_m[0:1, 2:3][0][0])
        cs_values_shap_impt.append(cs_m[1:2, 2:3][0][0])
    plt.plot(sorted(cs_values_lime_shap), color='blue', label='LIME - SHAP')
    plt.plot(sorted(cs_values_lime_impt), color='red', label='LIME - IMPT')
    plt.plot(sorted(cs_values_shap_impt), color='green', label='SHAP - IMPT')
    plt.legend()
    plt.show()

    # overall #####
    # exps_matrix_all = np.concatenate((exps['lime'].toarray(), exps['shap'].toarray(), exps['impt'].toarray()))
    # cs = cosine_similarity(exps_matrix_all)
    # plot_cs(cs)

    print('end')
