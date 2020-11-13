from _utils.explanation_loader import ExplanationLoader
from _utils.tweet_loader import TweetLoader
from _utils.trained_model_loader import TrainedModelLoader


class ExplainableTweet:
    """
    Attributes:
        - raw: original tweet
        - tokens: preprocessed tweet, tokens only
        - label: real label of the tweet (sexist or non-sexist, i.e. 1 or 0)
        - explanations (by every model): dict with explanation_method as key and explanation as value
        - predictions (by every model): dict with model_name as key and prediction as value

    Configuration:
        - k: explanation contains k most important words
    """
    k = 5

    def __init__(self, tweet_id):
        """
        input:
            tweet_id: place in test dataset
        """
        tweet_loader = TweetLoader()
        self.raw = tweet_loader.get_raw_tweet_from_id(tweet_id)
        self.tokens = tweet_loader.get_tweet_tokens_from_id(tweet_id)
        self.label = tweet_loader.get_tweet_label_from_id(tweet_id)

        xgboost_loader = ExplanationLoader('xgboost')
        svm_loader = ExplanationLoader('svm')
        lr_loader = ExplanationLoader('lr')

        self.explanations = {
            'xgboost': {
                'impt': xgboost_loader.get_explanation_from_method_by_tweet_id('impt', tweet_id, k=self.k),
                'lime': xgboost_loader.get_explanation_from_method_by_tweet_id('lime', tweet_id, k=self.k),
                'shap': xgboost_loader.get_explanation_from_method_by_tweet_id('shap', tweet_id, k=self.k)
            },
            'svm': {
                'impt': svm_loader.get_explanation_from_method_by_tweet_id('coef', tweet_id, k=self.k),
                'lime': svm_loader.get_explanation_from_method_by_tweet_id('lime', tweet_id, k=self.k),
                'shap': svm_loader.get_explanation_from_method_by_tweet_id('shap', tweet_id, k=self.k)
            },
            'lr': {
                'impt': lr_loader.get_explanation_from_method_by_tweet_id('impt', tweet_id, k=self.k),
                'lime': lr_loader.get_explanation_from_method_by_tweet_id('lime', tweet_id, k=self.k),
                'shap': lr_loader.get_explanation_from_method_by_tweet_id('shap', tweet_id, k=self.k)
            }
        }

        trained_model_loader = TrainedModelLoader()
        xgboost = trained_model_loader.load('xgboost')
        svm = trained_model_loader.load('svm')
        lr = trained_model_loader.load('lr')
        self.predictions = {
            'xgboost': xgboost.predict([self.raw])[0],
            'svm': svm.predict([self.raw])[0],
            'lr': lr.predict([self.raw])[0]
        }
