import os
import pickle
import numpy as np
import pandas as pd
import json
from _data.unsex_data import UnsexData
from scipy.stats import ttest_ind
from random import sample
from _utils.explanations_helper import load_features_and_scores
from time import time
import matplotlib.pyplot as plt
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
    @staticmethod
    def my_coefficient(exps, f):
        values_lime_shap = []
        values_lime_builtin = []
        values_shap_builtin = []
        for i in range(exps['lime'].shape[0]):
            if f is None:
                lime_array = exps['lime'].toarray()[i:i + 1, :]
                shap_array = exps['shap'].toarray()[i:i + 1, :]
                builtin_array = exps['builtin'].toarray()[i:i + 1, :]
                matrix = np.concatenate((lime_array, shap_array, builtin_array))
                cs_m = cosine_similarity(matrix)
                values_lime_shap.append(cs_m[0:1, 1:2][0][0])
                values_lime_builtin.append(cs_m[0:1, 2:3][0][0])
                values_shap_builtin.append(cs_m[1:2, 2:3][0][0])
            else:
                lime_array = exps['lime'].toarray()[i:i + 1, :][0]
                shap_array = exps['shap'].toarray()[i:i + 1, :][0]
                builtin_array = exps['builtin'].toarray()[i:i + 1, :][0]
                values_lime_shap.append(f(lime_array, shap_array))
                values_lime_builtin.append(f(lime_array, builtin_array))
                values_shap_builtin.append(f(shap_array, builtin_array))
        values_lime_shap = np.sort(values_lime_shap)
        values_lime_builtin = np.sort(values_lime_builtin)
        values_shap_builtin = np.sort(values_shap_builtin)
        return values_lime_shap, values_lime_builtin, values_shap_builtin


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
                fig = plt.figure()
                for i, experiment in enumerate(Experiments):
                    graph = fig.add_subplot(2, 2, i + 1)
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

                    graph.set_xlabel(f'${metric}\_coefficient$')
                    graph.set_ylabel('$p$')
                    if i + 1 in [1, 2]:
                        graph.axes.xaxis.set_visible(False)
                    if i + 1 in [2, 4]:
                        graph.axes.yaxis.set_visible(False)
                    # plot the CDF
                    y = np.array(range(len(values_lime_shap)))/float(len(values_lime_shap))
                    graph.plot(values_lime_shap, y, color='blue', label='LIME - SHAP')
                    graph.plot(values_lime_builtin, y, color='red', label='LIME - builtin')
                    graph.plot(values_shap_builtin, y, color='green', label='SHAP - builtin')
                    graph.set_title(f"{experiment.name} (F1: {get_f1_score(model, experiment)})", fontdict={'fontsize': 10})
                # fig.suptitle(f'{t(model)} (k={k}, measure={metric})', size=15)
                plt.legend()
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


def what_is_explainable(experiment, k=5):
    with open(os.path.join(get_experiment_path(experiment), 'used_data', 'X_test_raw.pkl'), 'rb') as f:
        all_raw = pickle.load(f)
    with open(os.path.join(get_experiment_path(experiment), 'used_data', 'X_test.pkl'), 'rb') as f:
        all_tokens = pickle.load(f)
    with open(os.path.join(get_experiment_path(experiment), f'explainable_tweets_k{k}.pkl'), 'rb') as f:
        explainable = pickle.load(f)
        explainable_raw = [et.raw for et in explainable]
        explainable_tokens = [et.tokens for et in explainable]
    not_explainable_raw = np.setdiff1d(all_raw, explainable_raw)
    not_explainable_tokens = np.setdiff1d(all_tokens, explainable_tokens)
    result = {}
    result['all_raw'] = {}
    result['all_raw']['amount'] = len(all_raw)
    result['explainable'] = {}
    result['explainable']['amount'] = len(explainable_raw)
    result['explainable']['min_tokens'] = min([len(t.split()) for t in explainable_tokens])
    result['explainable']['max_tokens'] = max([len(t.split()) for t in explainable_tokens])
    result['explainable']['min_raw_length'] = min([len(t) for t in explainable_raw])
    result['explainable']['max_raw_length'] = max([len(t) for t in explainable_raw])
    xs = np.array(range(len(explainable_tokens)))/len(explainable_tokens)
    exp_ys = sorted([len(t.split()) for t in explainable_tokens])
    plt.plot(xs, exp_ys, label='explainable')
    result['not_explainable'] = {}
    result['not_explainable']['amount'] = len(not_explainable_raw)
    result['not_explainable']['min_tokens'] = min([len(t.split()) for t in not_explainable_tokens])
    result['not_explainable']['max_tokens'] = max([len(t.split()) for t in not_explainable_tokens])
    result['not_explainable']['min_raw_length'] = min([len(t) for t in not_explainable_raw])
    result['not_explainable']['max_raw_length'] = max([len(t) for t in not_explainable_raw])
    xs = np.array(range(len(not_explainable_tokens)))/len(not_explainable_tokens)
    unexp_ys = sorted([len(t.split()) for t in not_explainable_tokens])
    plt.plot(xs, unexp_ys, label='unexplainable')
    plt.xlabel('$tweets$')
    plt.ylabel('$number\_of\_tokens$')
    plt.legend()
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'what_is_explainable.png'))
    # print(json.dumps(result, indent=1))
    print("P-Value: ", ttest_ind(exp_ys, unexp_ys).pvalue)  # 0.0007620476151910129


def local_to_global():
    start = time()
    ex_methods = ['builtin', 'shap', 'lime']
    models = ['lr', 'svm', 'xgboost']
    results = {}
    for experiment in Experiments:
        results[experiment.name] = {}
        for model in models:
            results[experiment.name][model] = {}
            for ex_method in ex_methods:
                global_exp = {}
                f, s = load_features_and_scores(model, ex_method, experiment)
                if ex_method == 'builtin':
                    for token in f:
                        global_exp[token] = 0
                    for token, value in zip(f, s):
                        global_exp[token] += value
                else:
                    for tweets in f:
                        for token in tweets:
                            global_exp[token] = 0
                    for tweets, values in zip(f, s):
                        for token, value in zip(tweets, values):
                            global_exp[token] += value
                results[experiment.name][model][ex_method] = list(dict(sorted(global_exp.items(), key=lambda i: i[1], reverse=True)).keys())[:20]
    # print(json.dumps(results, indent=1))
    print(f"Processing time: {int(time() - start)} seconds")
    for e in Experiments:
        _print_latex_global(e.name, results)


def _print_latex_global(e, results):
    x = results[e]
    experiment_name = e.replace('_', r'\_')
    a = list(zip(x['lr']['lime'], x['lr']['shap'], x['lr']['builtin'], x['svm']['lime'], x['svm']['shap'],
                 x['svm']['builtin'], x['xgboost']['lime'], x['xgboost']['shap'], x['xgboost']['builtin']))
    print(r"\begin{table}[!htbp]" + '\n'
          r"\centering" + '\n'
          r"\begin{adjustbox}{width=1\textwidth}" + '\n'
          r"\small" + '\n'
          r"\begin{tabular}{*3c|*3c|*3c}" + '\n'
          r"\toprule" + '\n'
          r"\multicolumn{9}{c}{\textbf{" + experiment_name + r"}}\\" + '\n'
          r"\multicolumn{3}{c}{\textit{LR}} & \multicolumn{3}{c}{\textit{SVM}} & \multicolumn{3}{c}{\textit{XGBoost}}\\" + '\n'
          r"LIME & SHAP & Built-in & LIME & SHAP & Built-in & LIME & SHAP & Built-in\\" + '\n'
          r"\midrule")
    for row in a:
        print(' & '.join(row) + '\\\\')
    print(r"\bottomrule" + '\n'
          r"\end{tabular}" + '\n'
          r"\end{adjustbox}" + '\n'
          r"\caption{" + f"Global explanations for the experiment '{experiment_name}'" + "}" + '\n'
          r"\end{table}" + '\n')


def print_latex_preprocessing_table():
    explainable_tweets = ExplanationLoader(Experiments.Train_Orig_Test_Orig).get_explainable_tweets()
    print(r"\begin{table}[!htbp]" + '\n'
          r"\centering" + '\n'
          r"\begin{tabular}{p{0.45\textwidth}p{0.45\textwidth}}" + '\n'
          r"\toprule" + '\n'
          r"Original Tweet & Preprocessed Tweet \\" + '\n'
          r"\midrule")
    for tweet in sample(explainable_tweets, k=10):
        print(' & '.join([tweet.raw.replace('#', '\#'), tweet.tokens]) + '\\\\')
        print(r'\hline')
    print(r"\bottomrule" + '\n'
          r"\end{tabular}" + '\n'
          r"\caption{" + f"Example original tweets and its preprocessed version" + "}" + '\n'
          r"\end{table}" + '\n')


if __name__ == '__main__':
    # create_similarity_graphs(models=['lr'], metrics=['cosine'], ks=[3])
    # create_similarity_graphs()
    # local_to_global()
    # create_explainable_tweets_hist()
    # what_is_explainable(Experiments.Train_Orig_Test_Orig)
    # UnsexData.plot_per_experiment()
    print_latex_preprocessing_table()
