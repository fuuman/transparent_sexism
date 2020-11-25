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


if __name__ == '__main__':
    models = ['lr', 'svm', 'xgboost']
    ex_methods = ['lime', 'shap', 'builtin']

    exps = {}
    for model in models:
        for k in [3, 5, 7]:
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

                # per ex_method #######
                cs_values_lime_shap = []
                cs_values_lime_builtin = []
                cs_values_shap_builtin = []
                for i in range(len(explainable_tweets)):
                    lime_array = exps['lime'].toarray()[i:i + 1, :]
                    shap_array = exps['shap'].toarray()[i:i + 1, :]
                    builtin_array = exps['builtin'].toarray()[i:i + 1, :]
                    matrix = np.concatenate((lime_array, shap_array, builtin_array))
                    cs_m = cosine_similarity(matrix)
                    cs_values_lime_shap.append(cs_m[0:1, 1:2][0][0])
                    cs_values_lime_builtin.append(cs_m[0:1, 2:3][0][0])
                    cs_values_shap_builtin.append(cs_m[1:2, 2:3][0][0])

                cs_values_lime_shap = np.sort(cs_values_lime_shap)
                cs_values_lime_builtin = np.sort(cs_values_lime_builtin)
                cs_values_shap_builtin = np.sort(cs_values_shap_builtin)
                # calculate the proportional values of samples
                p = 1. * np.arange(len(cs_values_lime_shap)) / (len(cs_values_lime_shap) - 1)
                graph.plot(cs_values_lime_shap, p, color='blue', label='LIME - SHAP')
                graph.plot(cs_values_lime_builtin, p, color='red', label='LIME - builtin')
                graph.plot(cs_values_shap_builtin, p, color='green', label='SHAP - builtin')
                graph.set_title(experiment.name, fontdict={'fontsize': 10})
            # plt.setp([a.get_xticklabels() for a in axs[0, :]], visible=False)
            # plt.setp([a.get_yticklabels() for a in axs[:, 1]], visible=False)
            fig.suptitle(f'{t(model)} (F1: {get_f1_score(model, experiment)}, k={k})', size=15)
            gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
            plt.legend()
            # plt.subplots_adjust(top=5)
            current_fig = plt.gcf()
            plt.show()
            current_fig.savefig(os.path.join(get_repo_path(), '_evaluation', 'graphs', f'{model}_k{k}_cosine_sim_cdf.png'))
