from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from my_work.unsex_data import UnsexData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle

nlp = spacy.load("en_core_web_sm")


def _tokenize(sentence):
    try:
        doc = nlp(sentence)
    except:
        print(type(sentence))
        print(sentence)
    return ' '.join([token.lemma_ for token in doc])


if __name__ == '__main__':
    ud = UnsexData()

    X, y = ud.get_raw_data()
    X = X.map(_tokenize)

    tf_idf_vec = TfidfVectorizer(max_features=5000)
    tf_idf_vec.fit(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    X_train = tf_idf_vec.transform(X_train)
    X_test = tf_idf_vec.transform(X_test)

    for reg in ['l1', 'l2']:
        if reg == 'l2':
            svm = LinearSVC()
        else:
            svm = LinearSVC(loss='squared_hinge', penalty=reg, dual=False)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        print(f'### SVM ({reg})')
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # save svm model trained on unsex data
        pickle.dump(svm, open(f'my_svm_{reg}.pkl', 'wb'))

    # testing
    # while 1:
    #     print('')
    #     i = input('Text: ')
    #     pred = svm.predict(tf_idf_vec.transform([_tokenize(i)]))[0]
    #     print('SEXISTISCH!!' if pred == 1 else 'Brav.')
    #     print('')

