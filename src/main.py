import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from loading_data import get_pan_12_data_doc2vec, get_balanced_data
import numpy as np

from rnn import train_rnn
from load_w2v import get_train_test_data_w2v

print(os.path.abspath("."))


def main():
    try:
        X = np.load('../dataset/X.npy')
        y = np.load('../dataset/y.npy')
    except FileNotFoundError:
        print("No data found on disk, loading from dataset please wait..")
        X, y = get_pan_12_data_doc2vec()
        np.save('../dataset/X.npy', X)
        np.save('../dataset/y.npy', y)
        print("Data successfully loaded")

    try:
        X_balanced = np.load('../dataset/X_balanced.npy')
        y_balanced = np.load('../dataset/y_balanced.npy')
    except FileNotFoundError:
        print("No balanced data found on disk, transforming please wait..")
        X_balanced, y_balanced = get_balanced_data(X, y)
        np.save('../dataset/X_balanced.npy', X_balanced)
        np.save('../dataset/y_balanced.npy', y_balanced)
        print("Data successfully transformed")

    try:
        X_w2vec = np.load('../dataset/X_w2vec.npy')
        y_w2vec = np.load('../dataset/y_w2vec.npy')
    except FileNotFoundError:
        print("No W2V data found on dist, transforming please wait..")
        X_w2vec, y_w2vec = get_train_test_data_w2v()
        np.save('../dataset/X_w2vec.npy', X_w2vec)
        np.save('../dataset/y_w2vec.npy', y_w2vec)
        print("W2V Data successfully transformed")

    try:
        X_w2vec_balanced = np.load('../dataset/X_w2vec_balanced.npy')
        y_w2vec_balanced = np.load('../dataset/y_w2vec_balanced.npy')
    except FileNotFoundError:
        print("No balanced W2V data found on dist, transforming please wait..")
        X_w2vec_balanced, y_w2vec_balanced = get_balanced_data(X_w2vec, y_w2vec)
        np.save('../dataset/X_w2vec_balanced.npy', X_w2vec_balanced)
        np.save('../dataset/y_w2vec_balanced.npy', y_w2vec_balanced)
        print("W2V balanced Data successfully transformed")

    print('Done with the loading phase of data')
    gnb_classifier = GaussianNB()
    lda_classifier = LinearDiscriminantAnalysis()
    qda_classifier = QuadraticDiscriminantAnalysis(store_covariance=True, )
    random_forest_classifier = RandomForestClassifier()
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    classifiers = [
        gnb_classifier,
        qda_classifier,
        lda_classifier,
        random_forest_classifier,
        knn_classifier
    ]

    # train_classifiers(X_w2vec_balanced, y_w2vec_balanced, classifiers)
    X_train, X_test, y_train, y_test = train_test_split(
        X_w2vec_balanced, y_w2vec_balanced, test_size=0.30, stratify=y_w2vec_balanced)
    train_rnn(X_train, y_train, max_len=500)
    # print("The follow ing output will be for smote data")
    # train_classifiers(X_balanced, y_balanced, classifiers)
    # train_rnn(X_balanced, y_balanced)


if __name__ == "__main__":
    main()
