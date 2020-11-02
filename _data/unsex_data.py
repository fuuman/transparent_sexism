import os
import pandas as pd
from _utils.pathfinder import get_repo_path
from sklearn.preprocessing import LabelEncoder
from _preprocessing.jha2017_preprocessing import preprocess_jha2017
from sklearn.model_selection import train_test_split


class UnsexData:
    def __init__(self):
        self.REPO_PATH = get_repo_path()
        self.DATA_PATH = os.path.join(self.REPO_PATH, '_data', 'icwsm2020_data')
        self.ALL_DATA_FILE_PATH = os.path.join(self.DATA_PATH, 'all_data.csv')
        # ALL_DATA_ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIR, 'all_data_annotations.csv')

        self.all_data = pd.read_csv(self.ALL_DATA_FILE_PATH, sep='\t')
        # self.all_data_annotations = pd.read_csv(self.ALL_DATA_ANNOTATIONS_FILE_PATH, sep='\t')

        # load data in dataframe
        df = pd.DataFrame(self.all_data)

        # filter adv examples
        df = df[df['of_id'].isnull()]

        X = df['text'].dropna().values.tolist()

        y = pd.DataFrame(df['sexist'].dropna())
        y['sexist'] = LabelEncoder().fit_transform(y['sexist'])
        y = y.to_numpy().ravel()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20)

    def get_raw_test_tweets(self):
        return self.X_test

    def get_preprocessed_data(self):
        # preprocessing by jha et al. 2017
        return list(preprocess_jha2017(self.X_train)), \
               list(preprocess_jha2017(self.X_test)), \
               self.y_train, \
               self.y_test

    def save_as_csv(self):
        X_train, X_test, y_train, y_test = self.get_preprocessed_data()
        # save training data as train.csv and test data as val.csv to train fast-bert
        # see: https://github.com/kaushaltrivedi/fast-bert
        df = pd.DataFrame(zip(X_train, y_train), columns=['text', 'label'])
        train_df = df.iloc[:2400, :]
        train_df.to_csv(os.path.join(self.REPO_PATH, '_data', 'as_csv', 'train.csv'))

        dev_df = df.iloc[2400:, :]
        dev_df.reset_index(drop=True, inplace=True)
        dev_df.to_csv(os.path.join(self.REPO_PATH, '_data', 'as_csv', 'dev.csv'))

        test_df = pd.DataFrame(zip(X_test, y_test), columns=['text', 'label'])
        test_df.to_csv(os.path.join(self.REPO_PATH, '_data', 'as_csv', 'val.csv'))
