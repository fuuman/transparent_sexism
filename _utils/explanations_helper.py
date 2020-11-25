import os
from _utils.pathfinder import get_experiment_path
import pickle


def load_features_and_scores(model, ex_method, experiment):
    path = os.path.join(get_experiment_path(experiment), 'explanations')
    if ex_method == 'builtin':
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
        # if 'bert' in model:
        #     with open(os.path.join(path, 'features', f'xgboost_lime_all_features.pkl'), 'rb') as f:
        #         tweet_count = len(pickle.load(f))
        #     _f = ['_' for _ in range(tweet_count)]
        #     _s = [[0] for _ in range(tweet_count)]
        #     tweet_ids = get_explainable_tweet_ids()
        #     for f, s, i in zip(features, scores, tweet_ids):
        #         _f[i] = f
        #         _s[i] = s
        #     features = _f
        #     scores = _s
    return list(map(lambda i: i.split(), features)), scores


def get_top_k_features(features, scores, k=None):
    comb = list(zip(features, scores))
    return [fe_sc[0] for fe_sc in sorted(comb, key=lambda i: i[1], reverse=True)][:k]


def get_explainable_tweet_ids(experiment):
    """
    - get the tweet ids of the tweets that have explanations from every method
    - ids are in relation to the X_test_raw.pkl dataset which contains 816 tweets
    """
    # filter "unimportant tweets" without non-zero LIME explanations
    # tweets with lime explanation that makes sense (!= [0,0,0,0,...])
    s1 = load_features_and_scores('xgboost', 'lime', experiment)[1]
    s2 = load_features_and_scores('lr', 'lime', experiment)[1]
    s3 = load_features_and_scores('svm', 'lime', experiment)[1]
    tweet_ids1 = [tweet_id for tweet_id, scores in enumerate(s1) if sum(scores) != 0]
    tweet_ids2 = [tweet_id for tweet_id, scores in enumerate(s2) if sum(scores) != 0]
    tweet_ids3 = [tweet_id for tweet_id, scores in enumerate(s3) if sum(scores) != 0]
    tweet_ids = list(
        set(tweet_ids1).intersection(
            set(tweet_ids2)).intersection(
            set(tweet_ids3))
    )
    return sorted(tweet_ids)
