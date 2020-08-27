import os
import pandas as pd


class UnsexData:
    DATA_PATH = os.path.join('icwsm2020_data')
    ALL_DATA_FILE_PATH = os.path.join(DATA_PATH, 'all_data.csv')
    ALL_DATA_ANNOTATIONS_FILE_PATH = os.path.join(DATA_PATH, 'all_data_annotations.csv')

    def __init__(self):
        self.all_data = pd.read_csv(self.ALL_DATA_FILE_PATH, sep='\t')
        # self.all_data_annotations = pd.read_csv(self.ALL_DATA_ANNOTATIONS_FILE_PATH, sep='\t')

    def get_preprocessed_data(self):
        return self.all_data['text'], self.all_data['sexist']
