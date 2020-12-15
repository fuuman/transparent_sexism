import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard
import json
from tqdm import tqdm
from itertools import combinations
from _data.unsex_data import UnsexData
from scipy.stats import mannwhitneyu
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
    'dice': lambda a, b: (2 * sum([i * j for i, j in zip(a, b)])) / (sum(a) + sum(b)),
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
    means = {}
    backup = {}
    for model in models:
        backup[model] = {}
        for metric in metrics:
            means[metric] = {}
            backup[model][metric] = {}
            for k in ks:
                backup[model][metric][k] = {}
                fig = plt.figure()
                means[metric][k] = []
                for i, experiment in enumerate(Experiments):
                    exps = {}
                    backup[model][metric][k][experiment] = {}
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

                    # backup
                    backup[model][metric][k][experiment]['values_lime_shap'] = values_lime_shap
                    backup[model][metric][k][experiment]['values_lime_builtin'] = values_lime_builtin
                    backup[model][metric][k][experiment]['values_shap_builtin'] = values_shap_builtin
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
                    f1 = get_f1_score(model, experiment)
                    graph.set_title(f"{experiment.name} (F1: {f1})", fontdict={'fontsize': 10})

                    means[metric][k].append(np.nanmean(values_lime_shap, dtype=np.float64))
                    means[metric][k].append(np.nanmean(values_lime_builtin, dtype=np.float64))
                    means[metric][k].append(np.nanmean(values_shap_builtin, dtype=np.float64))
                # fig.suptitle(f'{t(model)} (k={k}, measure={metric})', size=15)
                plt.legend()
                current_fig = plt.gcf()
                # plt.show()
                current_fig.savefig(
                    os.path.join(get_repo_path(), '_evaluation', 'graphs', metric, f'{model}_k{k}_{metric}_sim_cdf.png'))

    # values = sorted(means, key=lambda i: i[0])
    # xs = [i[0] for i in values]
    # ys = [i[1] for i in values]
    # plt.title(model)
    # plt.xlabel('F1-Score')
    # plt.ylabel(f'${metric}\_coefficient$')
    # plt.plot(xs, ys)
    # plt.show()
    pickle.dump(means, open(os.path.join(get_repo_path(), '_evaluation', 'means.pkl'), 'wb'))
    pickle.dump(backup, open(os.path.join(get_repo_path(), '_evaluation', 'similarity_backup.pkl'), 'wb'))


def plot_k_graph():
    with open(os.path.join(get_repo_path(), '_evaluation', 'means.pkl'), 'rb') as f:
        means = pickle.load(f)
    fig = plt.figure()
    for i, metric in enumerate(['cosine', 'dice', 'jaccard', 'overlap']):  # , 'raw']:
        graph = fig.add_subplot(2, 2, i + 1)
        xs = []
        ys = []
        for k in [3, 5, 7]:
            for v in means[metric][k]:
                xs.append(k)
                ys.append(v)
        graph.set_title(metric)
        graph.set_xlabel('$k$')
        graph.set_ylabel('similarity mean')
        graph.scatter(xs, ys)
        if i + 1 in [1, 2]:
            graph.axes.xaxis.set_visible(False)
        if i + 1 in [2, 4]:
            graph.axes.yaxis.set_visible(False)
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', f'plot_k_graph_{metric}.png'))


def plot_k_box():
    with open(os.path.join(get_repo_path(), '_evaluation', 'means.pkl'), 'rb') as f:
        means = pickle.load(f)
    for metric in ['cosine', 'dice', 'jaccard', 'overlap', 'raw']:
        # plt.title(metric)
        plt.xlabel('$k$')
        plt.ylabel('similarity mean')
        plt.boxplot(means[metric].values())
        plt.xticks([1, 2, 3], [3, 5, 7])
        current_fig = plt.gcf()
        plt.show()
        current_fig.savefig(
            os.path.join(get_repo_path(), '_evaluation', 'graphs', f'plot_k_{metric}_box.png'))


def amount_of_explainable_tweets():
    labels = [e.name for e in Experiments]
    values = [get_amount_of_explainable_tweets(e) for e in Experiments]
    plt.barh(labels, values, color='green')
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'amount_of_explainable_tweets.png'))


def which_length_is_explainable(k=5):
    fig = plt.figure()
    for i, experiment in enumerate(Experiments):
        graph = fig.add_subplot(2, 2, i + 1)
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
        graph.plot(xs, exp_ys, label='explainable')

        result['not_explainable'] = {}
        result['not_explainable']['amount'] = len(not_explainable_raw)
        result['not_explainable']['min_tokens'] = min([len(t.split()) for t in not_explainable_tokens])
        result['not_explainable']['max_tokens'] = max([len(t.split()) for t in not_explainable_tokens])
        result['not_explainable']['min_raw_length'] = min([len(t) for t in not_explainable_raw])
        result['not_explainable']['max_raw_length'] = max([len(t) for t in not_explainable_raw])
        xs = np.array(range(len(not_explainable_tokens)))/len(not_explainable_tokens)
        unexp_ys = sorted([len(t.split()) for t in not_explainable_tokens])

        ttest_p = round(ttest_ind(exp_ys, unexp_ys).pvalue, 4)
        mwu_p = round(mannwhitneyu(exp_ys, unexp_ys).pvalue, 4)
        print(f"\n{experiment.name} T-Test, P-Value: ", ttest_p)
        print(f"{experiment.name} Mann-Whitney U Test: P-Value: ", mwu_p)

        graph.set_xlabel(f'$tweets$')
        graph.set_ylabel('$number\_of\_tokens$')

        graph.text(0.48, 1.5, f'T-Test: p={ttest_p}\nMWU-Test: p={mwu_p}',
                   fontsize=8)

        if i + 1 in [1, 2]:
            graph.axes.xaxis.set_visible(False)
        if i + 1 in [2, 4]:
            graph.axes.yaxis.set_visible(False)

        graph.plot(xs, unexp_ys, label='unexplainable')
        graph.set_title(experiment.name, fontdict={'fontsize': 10})

    plt.legend()
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'which_length_is_explainable.png'))
    # print(json.dumps(result, indent=1))


def which_classes_are_explainable(k=5):
    labels = ['TOTO', 'TMTO', 'TOTM', 'TMTM']
    explainable_sexist = np.array([])
    explainable_non_sexist = np.array([])
    for experiment in Experiments:
        with open(os.path.join(get_experiment_path(experiment), f'explainable_tweets_k{k}.pkl'), 'rb') as f:
            explainable = pickle.load(f)
        explainable_sexist = np.append(explainable_sexist, len([e for e in explainable if e.label == 1]))
        explainable_non_sexist = np.append(explainable_non_sexist, len([e for e in explainable if e.label == 0]))
    explainable_sexist = explainable_sexist/(explainable_sexist + explainable_non_sexist)
    red, green = [plt.cm.Reds(0.5), plt.cm.Greens(0.4)]
    plt.bar(labels, [1] * len(explainable_sexist), color=green, label='Non-Sexist')
    plt.bar(labels, explainable_sexist, color=red, label='Sexist')
    plt.ylabel('Proportion of explainable tweets')
    plt.title('Which classes are explainable?')
    plt.legend()
    current_fig = plt.gcf()
    plt.show()
    # current_fig.savefig(
    #     os.path.join(get_repo_path(), '_evaluation', 'graphs', 'which_classes_are_explainable.png'))


def _get_dataset_of_tweet(explainable_tweet):
    ud = UnsexData(Experiments.Train_Orig_Test_Orig).all_data
    return ud.loc[ud['text'] == explainable_tweet.raw]['dataset'].values[0]


def _is_adversarial(explainable_tweet):
    ud = UnsexData(Experiments.Train_Orig_Test_Orig).all_data
    return ud.loc[ud['text'] == explainable_tweet.raw]['dataset'].values[0]


def which_datasets_are_explainable(k=5):
    labels = ['TOTO', 'TMTO', 'TOTM', 'TMTM']
    explainable_b = np.array([])
    explainable_h = np.array([])
    explainable_o = np.array([])
    explainable_c = np.array([])
    explainable_s = np.array([])
    for experiment in tqdm(Experiments):
        with open(os.path.join(get_experiment_path(experiment), f'explainable_tweets_k{k}.pkl'), 'rb') as f:
            explainable = pickle.load(f)
        explainable_b = np.append(explainable_b, len([e for e in tqdm(explainable) if _get_dataset_of_tweet(e) == 'benevolent']))
        explainable_h = np.append(explainable_h, len([e for e in tqdm(explainable) if _get_dataset_of_tweet(e) == 'hostile']))
        explainable_o = np.append(explainable_o, len([e for e in tqdm(explainable) if _get_dataset_of_tweet(e) == 'other']))
        explainable_c = np.append(explainable_c, len([e for e in tqdm(explainable) if _get_dataset_of_tweet(e) == 'callme']))
        explainable_s = np.append(explainable_s, len([e for e in tqdm(explainable) if _get_dataset_of_tweet(e) == 'scales']))
    explainable_b = explainable_b/len(explainable)
    explainable_h = explainable_h/len(explainable)
    explainable_o = explainable_o/len(explainable)
    explainable_c = explainable_c/len(explainable)
    explainable_s = explainable_s/len(explainable)

    c1, c2, c3, c4, c5 = plt.cm.Set1.colors[:5]
    plt.bar(labels, [1] * len(explainable_b), color=c1, label='benevolent')
    plt.bar(labels, explainable_h + explainable_o + explainable_c + explainable_s, color=c2, label='hostile')
    plt.bar(labels, explainable_o + explainable_c + explainable_s, color=c3, label='other')
    plt.bar(labels, explainable_c + explainable_s, color=c4, label='callme')
    plt.bar(labels, explainable_s, color=c5, label='scales')
    plt.ylabel('Proportion of explainable tweets')
    plt.title('Which datasets are explainable?')
    plt.legend()
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'which_datasets_are_explainable.png'))


def myjaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


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
    jaccard_for_global(results)
    # for e in Experiments:
    #     _print_latex_global(e.name, results)


def jaccard_for_global(results):
    for e in Experiments:
        a = results[e.name]
        models = ['lr', 'svm', 'xgboost']
        ex_m = ['lime', 'shap', 'builtin']
        ls = []
        for m in models:
            for em in ex_m:
                ls.append(a[m][em][:10])
        js = []
        for i in list(combinations(ls, 2)):
            js.append(myjaccard(i[0], i[1]))
        print(e, round(len([j for j in js if j >= 1/3])/len(js), 2))
        plt.plot(range(len(js)), sorted(js), label=e.name)
    plt.xlabel('global explanation')
    plt.ylabel('jaccard coefficient')
    plt.legend()
    plt.show()


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


def plot_f1_to_similarity():
    with open(os.path.join(get_repo_path(), '_evaluation', 'similarity_backup.pkl'), 'rb') as f:
        backup = pickle.load(f)
    xs = []
    ys = []
    for model in ['lr', 'svm', 'xgboost']:
        for experiment in Experiments:
            f1 = get_f1_score(model, experiment)
            all_values_lime_shap = []
            all_values_lime_builtin = []
            all_values_shap_builtin = []
            for k in [3, 5, 7]:
                for coef in ['cosine', 'dice', 'jaccard', 'overlap', 'raw']:
                    all_values_lime_shap.append(backup[model][coef][k][experiment]['values_lime_shap'])
                    all_values_lime_builtin.append(backup[model][coef][k][experiment]['values_lime_builtin'])
                    all_values_shap_builtin.append(backup[model][coef][k][experiment]['values_shap_builtin'])
            for sim_values in [all_values_lime_shap, all_values_lime_builtin, all_values_shap_builtin]:
                xs.append(f1)
                ys.append((np.nanmean(sim_values)))
    pearson_pvalue = format(pearsonr(xs, ys)[1], '.2f')
    spearman_pvalue = format(spearmanr(xs, ys).pvalue, '.2f')
    # print('Pearson: ', pearson_pvalue)
    # print('Spearman: ', spearmanr(xs, ys))
    plt.text(0.73, 0.12, f'Pearson: p={pearson_pvalue}\nSpearman: p={spearman_pvalue}', fontsize=10)
    plt.xlabel('F1-Score')
    plt.ylabel('Similarity Mean')
    plt.scatter(xs, ys)
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation', 'graphs', 'plot_f1_to_similarity.png'))


if __name__ == '__main__':
    start = time()

    # which_classes_are_explainable()
    # create_similarity_graphs()
    which_datasets_are_explainable()
    # means = create_similarity_graphs()
    # local_to_global()
    # amount_of_explainable_tweets()
    # which_length_is_explainable()
    # UnsexData.plot_per_experiment()
    # plot_k_box()
    # plot_f1_to_similarity()
    # plot_k_graph()
    # print_latex_preprocessing_table()

    print(f"Runtime: {round((time() - start)/60.0, 2)} min")
