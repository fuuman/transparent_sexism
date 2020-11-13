import os
from _utils.pathfinder import get_repo_path
import pickle


class TweetLoader:
    def __init__(self):
        self.tweets = self._load_test_tweets()
        self.raw_tweets = self._load_raw_test_tweets()
        self.labels = self._load_labels()

    @staticmethod
    def _load_raw_test_tweets():
        path = os.path.join(get_repo_path(), '_explanations', 'unsex')
        with open(os.path.join(path, 'used_training_data', 'X_test_raw.pkl'), 'rb') as f:
            tweets = pickle.load(f)
        return tweets

    @staticmethod
    def _load_test_tweets():
        path = os.path.join(get_repo_path(), '_explanations', 'unsex')
        with open(os.path.join(path, 'used_training_data', 'X_test.pkl'), 'rb') as f:
            tweets = pickle.load(f)
        return tweets

    @staticmethod
    def _load_labels():
        path = os.path.join(get_repo_path(), '_explanations', 'unsex')
        with open(os.path.join(path, 'used_training_data', 'y_test.pkl'), 'rb') as f:
            labels = pickle.load(f)
        return labels

    def get_raw_tweet_from_id(self, id):
        return self.raw_tweets[id]

    def get_tweet_tokens_from_id(self, id):
        return self.tweets[id]

    def get_tweet_label_from_id(self, id):
        return self.labels[id]
