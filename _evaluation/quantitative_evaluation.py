import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity
from _utils.pathfinder import get_experiment_path, get_repo_path
from _utils.experiments import Experiments
from _utils.tweet_loader import TweetLoader
from _utils.explanation_loader import ExplanationLoader
from _utils.explanations_helper import get_amount_of_explainable_tweets


def t(model_name):
    if model_name == 'lr':
        return 'LR'
    elif model_name == 'svm':
        return 'SVM'
    elif model_name == 'xgboost':
        return 'XGBoost'
    else:
        return 'Unknown Model'


def get_f1_score(model_name, experiment):
    df = pd.read_csv(os.path.join(get_experiment_path(experiment), 'reports', f'{model_name}.csv'))
    return round(df.iloc[4]['f1-score'], 2)


def transform(model_name, experiment, tokens):
    if model_name == 'svm':
        pipeline = pickle.load(open(os.path.join(get_experiment_path(experiment), 'models', f'{model_name}.pkl'), 'rb'),
                               encoding='latin1')
    else:
        pipeline = pickle.load(open(os.path.join(get_experiment_path(experiment), 'models', f'{model_name}.pkl'), 'rb'))
    tfidf = pipeline.named_steps['tfidf']
    return tfidf.transform(tokens)


class Metrics:
    # @staticmethod
    # def my_cosine_similarity(exps):
    #     cs_values_lime_shap = []
    #     cs_values_lime_builtin = []
    #     cs_values_shap_builtin = []
    #     for i in range(exps['lime'].shape[0]):
    #         lime_array = exps['lime'].toarray()[i:i + 1, :]
    #         shap_array = exps['shap'].toarray()[i:i + 1, :]
    #         builtin_array = exps['builtin'].toarray()[i:i + 1, :]
    #         matrix = np.concatenate((lime_array, shap_array, builtin_array))
    #         cs_m = cosine_similarity(matrix)
    #         cs_values_lime_shap.append(cs_m[0:1, 1:2][0][0])
    #         cs_values_lime_builtin.append(cs_m[0:1, 2:3][0][0])
    #         cs_values_shap_builtin.append(cs_m[1:2, 2:3][0][0])
    #
    #     cs_values_lime_shap = np.sort(cs_values_lime_shap)
    #     cs_values_lime_builtin = np.sort(cs_values_lime_builtin)
    #     cs_values_shap_builtin = np.sort(cs_values_shap_builtin)
    #     return cs_values_lime_shap, cs_values_lime_builtin, cs_values_shap_builtin

    @staticmethod
    def my_coefficient(exps, f):
        values_lime_shap = []
        values_lime_builtin = []
        values_shap_builtin = []
        for i in range(exps['lime'].shape[0]):
            lime_array = exps['lime'].toarray()[i:i + 1, :]
            shap_array = exps['shap'].toarray()[i:i + 1, :]
            builtin_array = exps['builtin'].toarray()[i:i + 1, :]
            if f is None:
                matrix = np.concatenate((lime_array, shap_array, builtin_array))
                cs_m = cosine_similarity(matrix)
                values_lime_shap.append(cs_m[0:1, 1:2][0][0])
                values_lime_builtin.append(cs_m[0:1, 2:3][0][0])
                values_shap_builtin.append(cs_m[1:2, 2:3][0][0])
            else:
                values_lime_shap.append(f(lime_array, shap_array))
                values_lime_builtin.append(f(lime_array, builtin_array))
                values_shap_builtin.append(f(shap_array, builtin_array))
        values_lime_shap = np.sort(values_lime_shap)
        values_lime_builtin = np.sort(values_lime_builtin)
        values_shap_builtin = np.sort(values_shap_builtin)
        return values_lime_shap, values_lime_builtin, values_shap_builtin

    # @staticmethod
    # def my_dice_coefficient(exps):
    #     values_lime_shap = []
    #     values_lime_builtin = []
    #     values_shap_builtin = []
    #     for i in range(exps['lime'].shape[0]):
    #         # 816 steps
    #         # in every step compare one tweet
    #         lime_array = exps['lime'].toarray()[i:i + 1, :][0]
    #         shap_array = exps['shap'].toarray()[i:i + 1, :][0]
    #         builtin_array = exps['builtin'].toarray()[i:i + 1, :][0]
    #
    #         dice = lambda a, b: (2 * sum([l * s for l, s in zip(a, b)])) / (sum(a) + sum(b))
    #
    #         values_lime_shap.append(dice(lime_array, shap_array))
    #         values_lime_builtin.append(dice(lime_array, builtin_array))
    #         values_shap_builtin.append(dice(shap_array, builtin_array))
    #     values_lime_shap = np.sort(values_lime_shap)
    #     values_lime_builtin = np.sort(values_lime_builtin)
    #     values_shap_builtin = np.sort(values_shap_builtin)
    #     return values_lime_shap, values_lime_builtin, values_shap_builtin
    #
    #
    # def my_jaccard_coefficient(exps):
    #     values_lime_shap = []
    #     values_lime_builtin = []
    #     values_shap_builtin = []
    #     for i in range(exps['lime'].shape[0]):
    #         # 816 steps
    #         # in every step compare one tweet
    #         lime_array = exps['lime'].toarray()[i:i + 1, :][0]
    #         shap_array = exps['shap'].toarray()[i:i + 1, :][0]
    #         builtin_array = exps['builtin'].toarray()[i:i + 1, :][0]
    #
    #         jaccard = lambda a, b: (sum([i * j for i, j in zip(a, b)])) / (
    #                     sum(a) + sum(b) - sum([i * j for i, j in zip(a, b)]))
    #
    #         values_lime_shap.append(jaccard(lime_array, shap_array))
    #         values_lime_builtin.append(jaccard(lime_array, builtin_array))
    #         values_shap_builtin.append(jaccard(shap_array, builtin_array))
    #     values_lime_shap = np.sort(values_lime_shap)
    #     values_lime_builtin = np.sort(values_lime_builtin)
    #     values_shap_builtin = np.sort(values_shap_builtin)
    #     return values_lime_shap, values_lime_builtin, values_shap_builtin
    #
    # def my_raw_coefficient(exps):
    #     values_lime_shap = []
    #     values_lime_builtin = []
    #     values_shap_builtin = []
    #     for i in range(exps['lime'].shape[0]):
    #         # 816 steps
    #         # in every step compare one tweet
    #         lime_array = exps['lime'].toarray()[i:i + 1, :][0]
    #         shap_array = exps['shap'].toarray()[i:i + 1, :][0]
    #         builtin_array = exps['builtin'].toarray()[i:i + 1, :][0]
    #
    #         jaccard = lambda a, b: sum([i * j for i, j in zip(a, b)])
    #
    #         values_lime_shap.append(jaccard(lime_array, shap_array))
    #         values_lime_builtin.append(jaccard(lime_array, builtin_array))
    #         values_shap_builtin.append(jaccard(shap_array, builtin_array))
    #     values_lime_shap = np.sort(values_lime_shap)
    #     values_lime_builtin = np.sort(values_lime_builtin)
    #     values_shap_builtin = np.sort(values_shap_builtin)
    #     return values_lime_shap, values_lime_builtin, values_shap_builtin
    #
    # def my_overlap_coefficient(exps):
    #     values_lime_shap = []
    #     values_lime_builtin = []
    #     values_shap_builtin = []
    #     for i in range(exps['lime'].shape[0]):
    #         # 816 steps
    #         # in every step compare one tweet
    #         lime_array = exps['lime'].toarray()[i:i + 1, :][0]
    #         shap_array = exps['shap'].toarray()[i:i + 1, :][0]
    #         builtin_array = exps['builtin'].toarray()[i:i + 1, :][0]
    #
    #         jaccard = lambda a, b: (sum([min(i, j) for i, j in zip(a, b)])) / (min(sum(a), sum(b)))
    #
    #         values_lime_shap.append(jaccard(lime_array, shap_array))
    #         values_lime_builtin.append(jaccard(lime_array, builtin_array))
    #         values_shap_builtin.append(jaccard(shap_array, builtin_array))
    #     values_lime_shap = np.sort(values_lime_shap)
    #     values_lime_builtin = np.sort(values_lime_builtin)
    #     values_shap_builtin = np.sort(values_shap_builtin)
    #     return values_lime_shap, values_lime_builtin, values_shap_builtin


METRIC_FUNCTIONS = {
    'cosine': None,
    'dice': lambda a, b: (2 * sum([l * s for l, s in zip(a, b)])) / (sum(a) + sum(b)),
    'jaccard': lambda a, b: (sum([i * j for i, j in zip(a, b)])) / (
                        sum(a) + sum(b) - sum([i * j for i, j in zip(a, b)])),
    'raw': lambda a, b: sum([i * j for i, j in zip(a, b)]),
    'overlap': lambda a, b: (sum([min(i, j) for i, j in zip(a, b)])) / (min(sum(a), sum(b)))
}


def create_similarity_graphs(models=None, metrics=None, ks=None):
    if models is None:
        models = ['lr', 'svm', 'xgboost']
    if metrics is None:
        metrics = ['cosine', 'dice', 'jaccard', 'raw', 'overlap']
    if ks is None:
        ks = [3, 5, 7]

    ex_methods = ['lime', 'shap', 'builtin']
    exps = {}
    for model in models:
        for metric in metrics:
            for k in ks:
                # fig, axs = plt.subplots(2, 2)
                fig = plt.gcf()
                gs = GridSpec(2, 2)
                axs = np.array([fig.add_subplot(ss) for ss in gs])
                # for graph, experiment in zip(axs.flat, Experiments):
                for graph, experiment in zip(axs, Experiments):
                    tweet_loader = TweetLoader(experiment)
                    explanation_loader = ExplanationLoader(experiment, tweet_loader=tweet_loader)
                    explainable_tweets = explanation_loader.get_explainable_tweets(k)

                    for ex_method in ex_methods:
                        exps[ex_method] = []
                        for explainable_tweet in explainable_tweets:
                            exps[ex_method].append(explainable_tweet.explanations[model][ex_method])
                        exps[ex_method] = np.array(list(map(lambda x: ' '.join(x), exps[ex_method])))
                        exps[ex_method] = transform(model, experiment, exps[ex_method])

                    ########## calculate metric ##########
                    values_lime_shap, values_lime_builtin, values_shap_builtin = Metrics.my_coefficient(exps, METRIC_FUNCTIONS[metric])
                    #####################################

                    # calculate the proportional values of samples
                    p = 1. * np.arange(len(values_lime_shap)) / (len(values_lime_shap) - 1)
                    graph.plot(values_lime_shap, p, color='blue', label='LIME - SHAP')
                    graph.plot(values_lime_builtin, p, color='red', label='LIME - builtin')
                    graph.plot(values_shap_builtin, p, color='green', label='SHAP - builtin')
                    graph.set_title(f"{experiment.name} (F1: {get_f1_score(model, experiment)})", fontdict={'fontsize': 10})
                # plt.setp([a.get_xticklabels() for a in axs[0, :]], visible=False)
                # plt.setp([a.get_yticklabels() for a in axs[:, 1]], visible=False)
                fig.suptitle(f'{t(model)} (k={k}, measure={metric})', size=15)
                gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
                plt.legend()
                # plt.subplots_adjust(top=5)
                current_fig = plt.gcf()
                plt.show()
                current_fig.savefig(
                    os.path.join(get_repo_path(), '_evaluation', 'graphs', metric, f'{model}_k{k}_{metric}_sim_cdf.png'))


def create_explainable_tweets_hist():
    labels = [e.name for e in Experiments]
    values = [get_amount_of_explainable_tweets(e) for e in Experiments]
    plt.barh(labels, values, color='green')
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'amount_of_explainable_tweets.png'))


if __name__ == '__main__':
    create_similarity_graphs(models=['lr'], metrics=['cosine'], ks=[3])
    # create_explainable_tweets_hist()
