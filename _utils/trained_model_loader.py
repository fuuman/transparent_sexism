from _utils.pathfinder import get_repo_path
import pickle
import os


class TrainedModelLoader:
    def __init__(self, experiment):
        self.experiment = experiment
        self.xgboost = self._load('xgboost')
        self.lr = self._load('lr')
        self.svm = self._load('svm')

    def _load(self, model_name):
        path = os.path.join(get_repo_path(), '_experiments', self.experiment.name, 'models', f'{model_name}.pkl')
        if 'svm' in model_name:
            model = pickle.load(open(path, 'rb'), encoding='latin1')
        else:
            model = pickle.load(open(path, 'rb'))
        return model
