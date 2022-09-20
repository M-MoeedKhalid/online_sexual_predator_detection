from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict

from helpers import get_values
from colorama import init
from colorama import Fore, Style
from datetime import datetime

init()


def train_classifiers(X, y, classifiers):
    k = 10
    kf = model_selection.KFold(n_splits=k, random_state=None)
    for classifier in classifiers:
        startTime = datetime.now()
        print(Fore.GREEN + f'The program has started training {classifier}' + Style.RESET_ALL)
        scores = cross_val_predict(classifier, X, y, cv=kf)
        conf_mat = confusion_matrix(y, scores)
        accuracy = accuracy_score(y, scores)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        auc = metrics.auc(fpr, tpr)
        precision, recall, f1 = get_values(conf_mat)
        print(f"Scores for {str(classifier)}")
        print(conf_mat)
        print(f'Accuracy : {accuracy}'
              f'\nPrecision: {precision}'
              f'\nRecall: {recall}'
              f'\nF1-score: {f1}'
              f'\nAUC: {auc}')
        print(f'Total time it took to run {classifier} was {datetime.now() - startTime}')
