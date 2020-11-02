from _training.train_models import train_all
from save_combinations import explain_all
from _data.unsex_data import UnsexData
import os
from _utils.pathfinder import get_repo_path
import pickle

if __name__ == '__main__':
    ud = UnsexData()

    X_train, X_test, y_train, y_test = ud.get_preprocessed_data()
    ud.save_as_csv()

    train_all(X_train, X_test, y_train, y_test)

    explain_all(X_train, X_test)

    # save used training data
    path = os.path.join(get_repo_path(), '_explanations', 'unsex', 'used_training_data')
    pickle.dump(X_train, open(os.path.join(path, 'X_train.pkl'), "wb"))
    pickle.dump(ud.get_raw_test_tweets(), open(os.path.join(path, 'X_test_raw.pkl'), "wb"))
    pickle.dump(y_train, open(os.path.join(path, 'y_train.pkl'), "wb"))
    pickle.dump(y_test, open(os.path.join(path, 'y_test.pkl'), "wb"))
