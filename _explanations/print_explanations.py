import os
import pickle
import utils
from _utils.pathfinder import get_repo_path
from _data.unsex_data import UnsexData


def _load_tweets():
    path = os.path.join(get_repo_path(), '_explanations', 'unsex')
    with open(os.path.join(path, 'tweets', 'tweets.pkl'), 'rb') as f:
        tweets = pickle.load(f)
    return tweets


def _load_features_and_scores(model, ex_method):
    path = os.path.join(get_repo_path(), '_explanations', 'unsex')
    if ex_method in ['impt', 'coef']:
        with open(os.path.join(path, 'features', f'{model}_{ex_method}_all_features.pkl'), 'rb') as f:
            c = pickle.load(f)
            features = ' '.join(c.keys())
            scores = list(c.values())
            return features, scores
    else:
        with open(os.path.join(path, 'features', f'{model}_{ex_method}_all_features.pkl'), 'rb') as f:
            features = pickle.load(f)
        with open(os.path.join(path, 'feature_importance', f'{model}_{ex_method}_all_scores.pkl'), 'rb') as f:
            scores = pickle.load(f)
        return features, scores


def _get_top_k_features(features, scores, k):
    comb = list(zip(features.split(), scores))
    return [fe_sc[0] for fe_sc in sorted(comb, key=lambda i: i[1], reverse=True)][:k]


def print_explanation(tweet_id, model_name, ex_methods=None, k=4):
    if ex_methods is None:
        ex_methods = ['lime', 'shap']
    # print(f'###### Explanations ({model_name}) ######')
    for ex_method in ex_methods:
        f, s = _load_features_and_scores(model_name, ex_method)
        print(f'{ex_method.upper()}: {_get_top_k_features(f[tweet_id], s[tweet_id], k)}')
    # print()


def print_prediction_and_label(tweet_id, model_name):
    classes = ['non-sexist', 'sexist']
    path = os.path.join(get_repo_path(), '_trained_models', f'{model_name}.pkl')
    if 'svm' in model_name:
        pipeline = utils.load_pickle(path, encoding=False)
    else:
        pipeline = utils.load_pickle(path)
    tweet = _load_tweets()[tweet_id]
    ud = UnsexData()
    tweets, labels = ud.get_preprocessed_data()
    label_idx = labels[tweets.index(tweet)]
    prediction = pipeline.predict([tweet])
    # print(f'###### Classes ######')
    print(f'Prediction of {model_name.upper()}: {classes[prediction[0]]}')
    # print(f'Real class: {classes[label_idx]}')


if __name__ == '__main__':
    # current options: svm, svm_l1, lr, xgboost
    model_name = 'xgboost'

    # current options: lime, shap
    explanation_method = 'lime'

    # filter "unimportant tweets" without non-zero LIME explanations
    s = _load_features_and_scores(model_name, explanation_method)[1]
    tweet_ids = [tweet_id for tweet_id, scores in enumerate(s) if sum(scores) != 0]
    tweets = _load_tweets()

    for i in tweet_ids[22:34]:
        print('###### Original Tweet ######')
        print(f'{tweets[i]}\n')
        print_explanation(i, model_name)
        print_prediction_and_label(i, model_name)

