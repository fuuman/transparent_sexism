from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from my_work.unsex_data import UnsexData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
import spacy
import pandas as pd
from fastbert import FastBERT
import os
from sklearn.model_selection import GridSearchCV
import pickle

nlp = spacy.load("en_core_web_sm")


def train(model, X_train, X_test, y_train, y_test, save_as=None, grid=False, report=None):
    """
    Args:
        model (Model object): model to train
        save_as (String): if set model will be saved with this string as filename after training
        grid:
        report (String): if set this method saves the classification report a csv file with that name

    Returns:
        classification report as pandas DataFrame (optional)
    """
    if 'fastbert' in str(type(model)):
        if save_as:
            model.fit(X_train, y_train, model_saving_path=os.path.join('models', f'{save_as}.bin'))
        else:
            model.fit(X_train, y_train)
        y_pred = []
        for tweet in X_test:
            y_pred.append(model(tweet, speed=0.7)[0])
    else:
        if grid:
            pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                                       ('classifier', model)])

            param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000],
                          'classifier__max_iter': [1000, 5000, 10000],
                          'tfidf__max_features': [None, 5000]}
            pipeline = GridSearchCV(pipeline, param_grid, scoring='accuracy', refit=True)
        else:
            pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english')), ('classifier', model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # save svm model trained on unsex data
        if save_as:
            pickle.dump(pipeline, open(os.path.join('models', f'{save_as}.pkl'), 'wb'))
            print(f'Model saved as {save_as}')

    if report:
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['non-sexist', 'sexist']))
        df = pd.DataFrame(classification_report(y_test, y_pred,
                                                target_names=['non-sexist', 'sexist'],
                                                output_dict=True)).transpose()
        df.to_csv(os.path.join('models', 'reports', f'{report}.csv'))


if __name__ == '__main__':
    ud = UnsexData()

    X, y = ud.get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print('###### SVM L1 ######')
    svm_l1 = LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    train(svm_l1, X_train, X_test, y_train, y_test, save_as='svm_l1', report='svm_l1')

    print('###### SVM L2 ######')
    svm = LinearSVC()
    train(svm, X_train, X_test, y_train, y_test, save_as='svm', report='svm')

    print('###### Logistic Regression ######')
    lr = LogisticRegression()
    train(lr, X_train, X_test, y_train, y_test, save_as='lr', report='lr')

    print('###### XGBoost ######')
    xgboost = XGBClassifier()
    train(xgboost, X_train, X_test, y_train, y_test, save_as='xgboost', report='xgboost')

    print('###### FastBERT ######')
    fastbert = FastBERT(
        kernel_name="google_bert_base_en",
        labels=[0, 1], device='cuda:0'
    )
    train(fastbert, X_train, X_test, y_train, y_test, save_as='fastbert', report='fastbert')

    print('Training finished.')
