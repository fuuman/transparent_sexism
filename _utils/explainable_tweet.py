from _utils.explanation_loader import ExplanationLoader
from _utils.tweet_loader import TweetLoader
from _utils.trained_model_loader import TrainedModelLoader


class ExplainableTweet:
    """
    Attributes:
        - experiment:       in which experimental setting was this tweet getting explained
        - raw:              original tweet
        - tokens:           preprocessed tweet, tokens only
        - label:            real label of the tweet (sexist or non-sexist, i.e. 1 or 0)
        - explanations (by every model):    dict with explanation_method as key and explanation as value
        - predictions (by every model):     dict with model_name as key and prediction as value
    """

    def __init__(self, tweet_id, experiment, k=5,
                 tweet_loader=None, explanation_loader=None, trained_model_loader=None):
        """
        input:
            tweet_id:       place in test dataset
            experiment:     experiment setting in which this tweet was used and explained
            k:              explanation contains k most important words
        """
        self.k = k
        self.experiment = experiment

        if tweet_loader is None:
            tweet_loader = TweetLoader(experiment)
        self.raw = tweet_loader.get_raw_tweet_from_id(tweet_id)
        self.tokens = tweet_loader.get_tweet_tokens_from_id(tweet_id)
        self.label = tweet_loader.get_tweet_label_from_id(tweet_id)

        if explanation_loader is None:
            explanation_loader = ExplanationLoader(experiment, tweet_loader=tweet_loader)

        self.explanations = {
            'xgboost': {
                'builtin': explanation_loader.get_explanation('xgboost', 'builtin', tweet_id, k=self.k),
                'lime': explanation_loader.get_explanation('xgboost', 'lime', tweet_id, k=self.k),
                'shap': explanation_loader.get_explanation('xgboost', 'shap', tweet_id, k=self.k)
            },
            'svm': {
                'builtin': explanation_loader.get_explanation('svm', 'builtin', tweet_id, k=self.k),
                'lime': explanation_loader.get_explanation('svm', 'lime', tweet_id, k=self.k),
                'shap': explanation_loader.get_explanation('svm', 'shap', tweet_id, k=self.k)
            },
            'lr': {
                'builtin': explanation_loader.get_explanation('lr', 'builtin', tweet_id, k=self.k),
                'lime': explanation_loader.get_explanation('lr', 'lime', tweet_id, k=self.k),
                'shap': explanation_loader.get_explanation('lr', 'shap', tweet_id, k=self.k)
            }
        }

        if trained_model_loader is None:
            trained_model_loader = TrainedModelLoader(experiment)
        self.predictions = {
            'xgboost': trained_model_loader.xgboost.predict([self.raw])[0],
            'svm': trained_model_loader.svm.predict([self.raw])[0],
            'lr': trained_model_loader.lr.predict([self.raw])[0]
        }

    def __str__(self):
        labels = ["non-sexist", "sexist"]
        string = f'--------------------\n' \
                 f'Original Tweet: {self.raw}\n' \
                 f'Preprocessed Tokens: {self.tokens}\n' \
                 f'Real Label: {labels[self.label]}\n' \
                 f'Predictions:\n'
        for model, prediction in self.predictions.items():
            string += f'{model.upper()} - {labels[prediction]}\n'
        for model, explanations in self.explanations.items():
            string += f'{model.upper()}:\n'
            for ex_method, explanation in explanations.items():
                string += f'{ex_method.upper()}: {explanation}\n'
        string += f'Experiment: {self.experiment}'
        string += '\n--------------------'
        return string
