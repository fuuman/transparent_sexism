from _utils.explanations_helper import load_features_and_scores
from _utils.explanations_helper import get_top_k_features
from _utils.tweet_loader import TweetLoader
from _utils.pathfinder import get_experiment_path
import os
import pickle


class ExplanationLoader:
    TOP_K_FROM_GLOBAL = 30

    def __init__(self, experiment, tweet_loader=None):
        self.experiment = experiment
        if tweet_loader is None:
            self.tl = TweetLoader(self.experiment)
        else:
            self.tl = tweet_loader
        self.features, self.scores = {}, {}
        for model in ['lr', 'svm', 'xgboost']:
            self.features[model] = {}
            self.scores[model] = {}
            for ex_method in ['builtin', 'lime', 'shap']:
                self.features[model][ex_method], self.scores[model][ex_method] = load_features_and_scores(model,
                                                                                                          ex_method,
                                                                                                          self.experiment)

    def get_explanation(self, model, ex_method, id, k=None):
        features, scores = self.features[model][ex_method], self.scores[model][ex_method]
        if ex_method == 'builtin':
            top_global_features = get_top_k_features(features, scores, self.TOP_K_FROM_GLOBAL)
            tweet = self.tl.get_tweet_tokens_from_id(id)
            exp = [f for f in top_global_features if f in tweet]
            if not exp:
                exp = ['']  # empty explanation if no global important words in the tweet
        else:
            # fs = [(f, s) for f, s in zip(f[tweet_id], s[tweet_id]) if f in tweet.split()]
            # f = [i[0] for i in fs]
            # s = [i[1] for i in fs]
            exp = get_top_k_features(features[id], scores[id], k)
        return exp

    def get_explainable_tweets(self, k=5):
        with open(os.path.join(get_experiment_path(self.experiment), f'explainable_tweets_k{k}.pkl'), 'rb') as f:
            explainable_tweets = pickle.load(f)
        return explainable_tweets
