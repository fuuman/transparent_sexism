from pathlib import Path

repo_path = str(Path(__file__).parent.parent.absolute())


def get_repo_path():
    return repo_path
