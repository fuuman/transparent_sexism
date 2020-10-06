import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class UnsexData:
    if os.name == 'posix':
        DATA_DIR = os.path.join('/run', 'media', 'marco', 'Daten', 'GitHub',
                                'feature-importance', 'my_work', 'icwsm2020_data')
    else:
        DATA_DIR = os.path.join('D:\\', 'GitHub', 'feature-importance', 'my_work', 'icwsm2020_data')
    ALL_DATA_FILE_PATH = os.path.join(DATA_DIR, 'all_data.csv')
    ALL_DATA_ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIR, 'all_data_annotations.csv')

    def __init__(self):
        self.all_data = pd.read_csv(self.ALL_DATA_FILE_PATH, sep='\t')
        # self.all_data_annotations = pd.read_csv(self.ALL_DATA_ANNOTATIONS_FILE_PATH, sep='\t')

    def get_preprocessed_data(self):
        # load data in dataframe
        df = pd.DataFrame(self.all_data)

        # filter adv examples
        df = df[df['of_id'].isnull()]

        X = df['text'].dropna().values.tolist()
        y = pd.DataFrame(df['sexist'].dropna())
        y['sexist'] = LabelEncoder().fit_transform(y['sexist'])
        y = y.to_numpy().ravel()
        return X, y

    @staticmethod
    def save_as_csv(X_train, X_test, y_train, y_test):
        # save training data as train.csv and test data as val.csv to train fast-bert
        # see: https://github.com/kaushaltrivedi/fast-bert
        train_df = pd.DataFrame(zip(X_train, y_train), columns=['text', 'label'])
        train_df.to_csv(os.path.join('csv', 'train.csv'))

        test_df = pd.DataFrame(zip(X_test, y_test), columns=['text', 'label'])
        train_df.to_csv(os.path.join('csv', 'val.csv'))
