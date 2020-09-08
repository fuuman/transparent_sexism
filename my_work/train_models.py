from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import ColumnSelector
from my_work.unsex_data import UnsexData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
import spacy
import os
from sklearn.model_selection import GridSearchCV
import pickle

nlp = spacy.load("en_core_web_sm")


# class SelectColumnsTransformer:
#     def __init__(self, columns=None):
#         self.columns = columns
#
#     def transform(self, X, **transform_params):
#         cpy_df = X[self.columns].copy()
#         return cpy_df
#
#     def fit(self, X, y=None, **fit_params):
#         return self


def train(model, save_as=None, grid=False):
    ud = UnsexData()

    X, y = ud.get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if grid:
        pipeline = Pipeline(steps=[('col_selector', ColumnSelector(cols=['text'], drop_axis=True)),
                                   ('tfidf', TfidfVectorizer(max_features=5000)),
                                   ('classifier', model)])

        param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000],
                      'classifier__max_iter': [1000, 5000, 10000],
                      'tfidf__max_features': [None, 5000]}
        svm_model = GridSearchCV(pipeline, param_grid, scoring='accuracy', refit=True)
    else:
        svm_model = Pipeline(steps=[('col_selector', ColumnSelector(cols=['text'], drop_axis=True)),
                                    ('tfidf', TfidfVectorizer(max_features=5000)),
                                    ('classifier', model)])

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['non-sexist', 'sexist']))

    # save svm model trained on unsex data
    if save_as:
        pickle.dump(svm_model, open(os.path.join('models', f'{save_as}.pkl'), 'wb'))
        print(f'Model saved as {save_as}')


if __name__ == '__main__':
    print('###### SVM L1 ######')
    svm_l1 = LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    train(svm_l1, save_as='svm_l1')

    print('###### SVM L2 ######')
    svm_l2 = LinearSVC()
    train(svm_l2, save_as='svm')

    print('###### Logistic Regression ######')
    lr = LogisticRegression()
    train(lr, save_as='lr')

    print('###### XGBoost ######')
    xgboost = XGBClassifier()
    train(xgboost, save_as='xgboost')
