from _utils.pathfinder import get_experiment_path, get_repo_path
from _utils.experiments import Experiments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def _get_f1_score(model_name, experiment):
    df = pd.read_csv(os.path.join(get_experiment_path(experiment), 'reports', f'{model_name}.csv'))
    return round(df.iloc[4]['f1-score'], 2)


def _get_accuracy(model_name, experiment):
    df = pd.read_csv(os.path.join(get_experiment_path(experiment), 'reports', f'{model_name}.csv'))
    return round(df.iloc[2]['precision'], 2)


def plot_model_performance(f):
    metric = None
    if 'accuracy' in f.__name__:
        metric = 'Accuracy'
    elif 'f1' in f.__name__:
        metric = 'F1-Score'
    models = ['lr', 'svm', 'xgboost', 'fast-bert']
    experiment_names = ['TOTO', 'TMTO', 'TOTM', 'TMTM']
    all_f1s = np.array([0] * len(experiment_names))
    for model in models:
        values = [f(model, experiment) for experiment in Experiments]
        all_f1s = all_f1s + np.array(values)
        plt.plot(experiment_names, values, label=model)
        for x, y in zip(experiment_names, values):
            plt.annotate("{:.2f}".format(y),  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
    plt.plot(experiment_names, all_f1s / len(models), label='mean')
    plt.legend()
    plt.xlabel('$experiment$')
    plt.ylabel(metric)
    current_fig = plt.gcf()
    plt.show()
    current_fig.savefig(
        os.path.join(get_repo_path(), '_evaluation',
                     'graphs', f'model_performance_{metric.lower().replace("-", "_")}.png'))


if __name__ == '__main__':
    plot_model_performance(_get_f1_score)
    plot_model_performance(_get_accuracy)
