import os
import pandas as pd
from _utils.pathfinder import get_repo_path
from sklearn.preprocessing import LabelEncoder
from _preprocessing.jha2017_preprocessing import preprocess_jha2017
from sklearn.model_selection import train_test_split
from _utils.experiments import Experiments


class UnsexData:
    def __init__(self, experiment):
        self.experiment = experiment
        self.REPO_PATH = get_repo_path()
        self.DATA_PATH = os.path.join(self.REPO_PATH, '_data', 'icwsm2020_data')
        self.ALL_DATA_FILE_PATH = os.path.join(self.DATA_PATH, 'all_data.csv')
        # ALL_DATA_ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIR, 'all_data_annotations.csv')

        self.all_data = pd.read_csv(self.ALL_DATA_FILE_PATH, sep='\t')
        # self.all_data_annotations = pd.read_csv(self.ALL_DATA_ANNOTATIONS_FILE_PATH, sep='\t')

        # load data in dataframe
        orig_mods_df = pd.DataFrame(self.all_data)
        orig_mods_df.dropna(axis=0, subset=['sexist'], inplace=True)  # drop NAs
        orig_df = orig_mods_df[orig_mods_df['of_id'].isnull()]
        mods_df = orig_mods_df[orig_mods_df['of_id'].notnull()]
        MAX_DATA_SIZE = len(orig_df)

        # proportions
        TRAIN_SIZE = round(MAX_DATA_SIZE * 0.8)  # 3262
        TEST_SIZE = round(MAX_DATA_SIZE * 0.2)  # 816
        TRAIN_SPLIT_ORG = round(TRAIN_SIZE * 0.65)  # 2120
        TRAIN_SPLIT_MOD = round(TRAIN_SIZE - TRAIN_SPLIT_ORG)  # 1142
        TEST_SPLIT_ORG = round(TEST_SIZE * 0.65)  # 530
        TEST_SPLIT_MOD = round(TEST_SIZE - TEST_SPLIT_ORG)  # 286

        if experiment == Experiments.ORIG_AND_ORIG:
            # train: 3262 orig
            # test: 816 orig
            self.X = orig_df['text'].values.tolist()

            y = pd.DataFrame(orig_df['sexist'])
            y['sexist'] = LabelEncoder().fit_transform(y['sexist'])
            self.y = y.to_numpy().ravel()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20)
        elif experiment == Experiments.ORIG_AND_ORIG_MODS:
            # train: 3262 orig
            # test: 530 orig + 286 mods

            # train
            train_df = orig_df.sample(TRAIN_SIZE)

            # test
            test_org_df = orig_df[~orig_df.isin(train_df)].dropna(how='all').sample(TEST_SPLIT_ORG)
            test_mod_df = mods_df.sample(TEST_SPLIT_MOD)
            test_df = test_org_df.append(test_mod_df, ignore_index=True)

            self.X_train, self.X_test, self.y_train, self.y_test = self.process(train_df, test_df)
        elif experiment == Experiments.ORIG_MODS_AND_ORIG:
            # train: 2120 orig + 1142 mods
            # test: 816 orig

            # test
            test_df = orig_df.sample(TEST_SIZE)

            # train
            train_org_df = orig_df[~orig_df.isin(test_df)].dropna(how='all').sample(TRAIN_SPLIT_ORG)
            train_mod_df = mods_df.sample(TRAIN_SPLIT_MOD)
            train_df = train_org_df.append(train_mod_df, ignore_index=True)

            self.X_train, self.X_test, self.y_train, self.y_test = self.process(train_df, test_df)
        elif experiment == Experiments.ORIG_MODS_AND_ORIG_MODS:
            # train: 2120 orig + 1142 mods
            # test: 530 orig + 286 mods

            # train
            train_org_df = orig_df.sample(TRAIN_SPLIT_ORG)
            train_mod_df = mods_df.sample(TRAIN_SPLIT_MOD)
            train_df = train_org_df.append(train_mod_df, ignore_index=True)

            # test
            test_org_df = orig_df[~orig_df.isin(train_org_df)].dropna(how='all').sample(TEST_SPLIT_ORG)
            test_mod_df = mods_df[~mods_df.isin(train_mod_df)].dropna(how='all').sample(TEST_SPLIT_MOD)
            test_df = test_org_df.append(test_mod_df, ignore_index=True)

            self.X_train, self.X_test, self.y_train, self.y_test = self.process(train_df, test_df)
        else:
            raise ValueError('Unknown Experiment')

    def get_all_data(self):
        return self.X, self.y

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

    @staticmethod
    def process(train_df, test_df):
        X_train = train_df['text'].values.tolist()
        y = pd.DataFrame(train_df['sexist'])
        y['sexist'] = LabelEncoder().fit_transform(y['sexist'])
        y_train = y.to_numpy().ravel()
        X_test = test_df['text'].values.tolist()
        y = pd.DataFrame(test_df['sexist'])
        y['sexist'] = LabelEncoder().fit_transform(y['sexist'])
        y_test = y.to_numpy().ravel()
        return X_train, X_test, y_train, y_test
