from pathlib import Path
import os

repo_path = str(Path(__file__).parent.parent.absolute())


def get_repo_path():
    return repo_path


def get_experiment_path(experiment):
    return os.path.join(get_repo_path(), '_experiments', experiment.name)
