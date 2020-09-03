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
import pickle

nlp = spacy.load("en_core_web_sm")


def train(model):
    ud = UnsexData()

    X, y = ud.get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    svm_model = Pipeline(steps=[('col_selector', ColumnSelector(cols=['text'], drop_axis=True)),
                                ('tfidf', TfidfVectorizer(max_features=5000)),
                                ('classifier', model)])
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    print('###### SVM L1 ######')
    svm_l1 = LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    train(svm_l1)

    print('###### SVM L2 ######')
    svm_l2 = LinearSVC()
    train(svm_l2)

    print('###### Logistic Regression ######')
    lr = LogisticRegression()
    train(lr)

    print('###### XGBoost ######')
    xgboost = XGBClassifier()
    train(xgboost)


# save svm model trained on unsex data
# pickle.dump(svm_model, open(f'my_svm_{reg}.pkl', 'wb'))

# testing
# while 1:
#     print('')
#     i = input('Text: ')
#     pred = svm.predict(tf_idf_vec.transform([_tokenize(i)]))[0]
#     print('SEXISTISCH!!' if pred == 1 else 'Brav.')
#     print('')

