from _training.train_models import train_all
from save_combinations import explain_all
from _utils.explainable_tweet import ExplainableTweet
from _data.unsex_data import UnsexData
import os
from tqdm import tqdm
from _utils.tweet_loader import TweetLoader
from _utils.explanation_loader import ExplanationLoader
from _utils.trained_model_loader import TrainedModelLoader
import logging
from _utils.pathfinder import get_experiment_path
import pickle
from datetime import datetime
from _utils.experiments import Experiments

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    ks = [3, 5, 7]
    number_of_experiments = len(Experiments)
    X_test = [1] * 816
    logging.info(f'Start pipeline at {datetime.now()}')
    for i, experiment in enumerate(Experiments):
        # logging.info(f"# Start Experiment {i + 1}/{number_of_experiments}: {experiment.name} #")
        # ud = UnsexData(experiment)
        #
        # logging.info(f'Experiment {i + 1}/{number_of_experiments}: Load data..')
        # X_train, X_test, y_train, y_test = ud.get_preprocessed_data()
        # # ud.save_as_csv()
        #
        # logging.info(f'Experiment {i + 1}/{number_of_experiments}: Start training..')
        # train_all(X_train, X_test, y_train, y_test, experiment=experiment)
        #
        # logging.info(f'Experiment {i + 1}/{number_of_experiments}: Start explaining..')
        # explain_all(X_train, X_test, experiment=experiment)
        #
        # # save used training data
        # logging.info(f'Experiment {i + 1}/{number_of_experiments}: Saving used data..')
        # path = os.path.join(get_experiment_path(experiment), 'used_data')
        # pickle.dump(X_train, open(os.path.join(path, 'X_train.pkl'), "wb"))
        # pickle.dump(X_test, open(os.path.join(path, 'X_test.pkl'), "wb"))
        # pickle.dump(ud.get_raw_test_tweets(), open(os.path.join(path, 'X_test_raw.pkl'), "wb"))
        # pickle.dump(y_train, open(os.path.join(path, 'y_train.pkl'), "wb"))
        # pickle.dump(y_test, open(os.path.join(path, 'y_test.pkl'), "wb"))

        # create pickle with ExplainableTweets for all tweets from the explained test set
        for k in ks:
            logging.info(f'Experiment {i + 1}/{number_of_experiments}: Saving ExplainableTweets..')

            tweet_loader = TweetLoader(experiment)
            explanation_loader = ExplanationLoader(experiment, tweet_loader=tweet_loader)
            trained_model_loader = TrainedModelLoader(experiment)

            explainable_tweets = [ExplainableTweet(tweet_id=tweet_id, experiment=experiment,
                                                   tweet_loader=tweet_loader, explanation_loader=explanation_loader,
                                                   trained_model_loader=trained_model_loader, k=k)
                                  for tweet_id in tqdm(range(len(X_test)))]
            pickle.dump(explainable_tweets,
                        open(os.path.join(get_experiment_path(experiment), f'explainable_tweets_k{k}.pkl'), "wb"))

    logging.info(f'Finished pipeline at {datetime.now()}')
