# Loading the libraries
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def get_pan_12_data():
    etree = et.parse('../dataset/pan12.xml')
    xroot = etree.getroot()
    predators = []
    file = open('../dataset/pan12_predators_list.txt', "r")
    for line in file:
        predators.append(line.strip())

    df_cols = ["id", "line", "author", "time", "text", "label"]
    rows = []

    for node in xroot:
        s_ids = node.attrib.get("id")
        for x in node:
            s_line = x.attrib.get("line") if node is not None else None
            s_author = x.find("author").text if node is not None else None
            s_time = x.find("time").text if node is not None else None
            s_text = x.find("text").text if node is not None else None
            rows.append({"id": s_ids, "line": s_line,
                         "author": s_author, "time": s_time, "text": s_text,
                         "label": 1 if s_author in predators else 0})

    df = pd.DataFrame(rows, columns=df_cols)
    df = df.dropna()

    df['id'].value_counts()
    df[['id', 'label']].value_counts()
    dataset = df['text']
    data = [d for d in dataset]

    def tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    data_for_training = list(tagged_document(data))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=5, )
    model.build_vocab(data_for_training)
    model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)
    result = []
    for i in range(0, len(model.docvecs)):
        if i <= (len(model.docvecs) - 1):
            result.append(model.docvecs[i])
        else:
            break

    Result = pd.DataFrame(result)
    nlp_result = np.array(Result)
    df["vector_model"] = result
    X = nlp_result
    y = np.array(df["label"], dtype='int')

    return X, y


def get_balanced_data(X, y):
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote
