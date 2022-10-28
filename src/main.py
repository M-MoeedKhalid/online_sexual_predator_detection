import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from execute_classifiers import train_classifiers
from loading_data import get_pan_12_data, get_balanced_data
import numpy as np

from src.rnn import train_rnn

print(os.path.abspath("."))


def main():
    try:
        X = np.load('../dataset/X.npy')
        y = np.load('../dataset/y.npy')
    except FileNotFoundError:
        print("No data found on disk, loading from dataset please wait..")
        X, y = get_pan_12_data()
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

    # train_classifiers(X, y, classifiers)
    train_rnn(X, y)
    print("The follow ing output will be for smote data")
    # train_classifiers(X_balanced, y_balanced, classifiers)
    train_rnn(X_balanced, y_balanced)


if __name__ == "__main__":
    main()
