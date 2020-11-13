from _utils.pathfinder import get_repo_path
import pickle
import os


class TrainedModelLoader:
    @staticmethod
    def load(model_name):
        path = os.path.join(get_repo_path(), '_trained_models', f'{model_name}.pkl')
        if 'svm' in model_name:
            model = pickle.load(open(path, 'rb'), encoding='latin1')
        else:
            model = pickle.load(open(path, 'rb'))
        return model
