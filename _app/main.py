from _training.train_models import train_all
from save_combinations import explain_all
from _data.unsex_data import UnsexData
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ud = UnsexData()

    X, y = ud.get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    ud.save_as_csv(X_train, X_test, y_train, y_test)

    train_all(X_train, X_test, y_train, y_test)

    explain_all(X_train, X_test)
