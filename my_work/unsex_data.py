import os
import pandas as pd


class UnsexData:
    REPO_DIR = os.path.dirname(os.path.abspath('my_work'))
    MY_WORK_PATH = os.path.join(REPO_DIR, 'my_work')
    DATA_PATH = os.path.join(MY_WORK_PATH, 'icwsm2020_data')
    ALL_DATA_FILE_PATH = os.path.join(DATA_PATH, 'all_data.csv')
    ALL_DATA_ANNOTATIONS_FILE_PATH = os.path.join(DATA_PATH, 'all_data_annotations.csv')

    def __init__(self):
        self.all_data = pd.read_csv(self.ALL_DATA_FILE_PATH, sep='\t')
        # self.all_data_annotations = pd.read_csv(self.ALL_DATA_ANNOTATIONS_FILE_PATH, sep='\t')

    def get_raw_data(self):
        return self.all_data['text'].dropna(), self.all_data['sexist'].dropna()
