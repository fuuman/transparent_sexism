from _utils.explanations_helper import load_features_and_scores
from _utils.explanations_helper import get_top_k_features
from _utils.tweet_loader import TweetLoader


class ExplanationLoader:
    TOP_K_FROM_GLOBAL = 30

    def __init__(self, model):
        self.model = model

    def get_explanation_from_method_by_tweet_id(self, ex_method, id, k=None):
        features, scores = load_features_and_scores(self.model, ex_method)
        if ex_method in ['coef', 'impt']:
            top_global_features = get_top_k_features(features, scores, self.TOP_K_FROM_GLOBAL)
            tl = TweetLoader()
            tweet = tl.get_tweet_tokens_from_id(id)
            exp = [f for f in top_global_features if f in tweet]
            # if not exp:
            # exp = ['']  # empty explanation if no global important words in the tweet
        else:
            # fs = [(f, s) for f, s in zip(f[tweet_id], s[tweet_id]) if f in tweet.split()]
            # f = [i[0] for i in fs]
            # s = [i[1] for i in fs]
            exp = get_top_k_features(features[id], scores[id], k)
        return exp
