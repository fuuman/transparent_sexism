from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from my_work.unsex_data import UnsexData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
import spacy
import os
from sklearn.model_selection import GridSearchCV
import pickle

nlp = spacy.load("en_core_web_sm")


def train(model, save_as=None, grid=False):
    ud = UnsexData()

    X, y = ud.get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if grid:
        pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                                   ('classifier', model)])

        param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000],
                      'classifier__max_iter': [1000, 5000, 10000],
                      'tfidf__max_features': [None, 5000]}
        pipeline = GridSearchCV(pipeline, param_grid, scoring='accuracy', refit=True)
    else:
        pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=5000)), ('classifier', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['non-sexist', 'sexist']))

    # save svm model trained on unsex data
    if save_as:
        pickle.dump(pipeline, open(os.path.join('models', f'{save_as}.pkl'), 'wb'))
        print(f'Model saved as {save_as}')


if __name__ == '__main__':
    print('###### SVM L1 ######')
    svm_l1 = LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    train(svm_l1, save_as='svm_l1')

    print('###### SVM L2 ######')
    svm = LinearSVC()
    train(svm, save_as='svm')

    print('###### Logistic Regression ######')
    lr = LogisticRegression()
    train(lr, save_as='lr')

    print('###### XGBoost ######')
    xgboost = XGBClassifier()
    train(xgboost, save_as='xgboost')
