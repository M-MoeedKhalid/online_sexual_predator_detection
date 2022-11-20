from gensim.models import Word2Vec
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

ROW_LENGTH = 5


def str2vec(str, model):
    tokens = str.split()
    vector = []
    for index, token in enumerate(tokens):
        try:
            w2v_default_value = model.wv[tokens[0]]
        except Exception:
            try:
                w2v_default_value = model.wv[tokens[1]]
            except:
                try:
                    w2v_default_value = model.wv[tokens[2]]
                except:
                    try:
                        w2v_default_value = model.wv[tokens[3]]
                    except:
                        try:
                            w2v_default_value = model.wv[tokens[4]]
                        except:
                            w2v_default_value = []
        except:
            w2v_default_value = []

        try:
            if token == '<et>':
                vec = w2v_default_value
            else:
                vec = model.wv[token]
            vector.extend(vec)
        except:
            if len(w2v_default_value) != 0:
                # for v in w2v_default_value:
                vector.extend(w2v_default_value)

    # vector = [item for sublist in vector for item in sublist]
    return vector


def get_multiple_rows(row):
    tokens = ' '.join(row['text'].split()).split()
    # tokens = row['text']
    n_word_string = ''
    list_of_rows = []
    for index, token in enumerate(tokens):
        if (index + 1) % ROW_LENGTH != 0:
            n_word_string = n_word_string + " " + token
        else:
            n_word_string = n_word_string + " " + token
            row['text'] = n_word_string
            list_of_rows.append(row.copy())
            n_word_string = ''
    if n_word_string != '':
        curr_length = len(n_word_string.strip().split(' '))
        if curr_length < ROW_LENGTH:
            for i in range(ROW_LENGTH - curr_length):
                n_word_string = n_word_string + ' <et>'
        row['text'] = n_word_string
        list_of_rows.append(row.copy())
    return list_of_rows


def get_train_test_data_w2v():
    w2v = Word2Vec.load('../dataset/word2vec.model')
    predators = []
    file = open('../dataset/pan12_predators_list.txt', "r")
    for line in file:
        predators.append(line.strip())
    dict_list = []
    rows_list = []
    cols = ['author', 'time', 'text', 'ispredator']
    tree = ET.parse('../dataset/pan12.xml')
    counter = 0
    row = dict()

    for elt in tree.iter():
        if elt.tag in cols:
            row[elt.tag] = elt.text
            if elt.tag == 'author':
                if elt.text in predators:
                    row['ispredator'] = 1
                else:
                    row['ispredator'] = 0

        if elt.tag == 'message':
            if row and row['text']:
                row['og_text'] = row['text']
                row_length = len(row['text'].split())
                if row_length > ROW_LENGTH:
                    # continue
                    list_of_rows = get_multiple_rows(row)
                    for r in list_of_rows:
                        rows_list.append(r)
                elif row_length < ROW_LENGTH:
                    # continue
                    for i in range(ROW_LENGTH - row_length):
                        row['text'] = row['text'] + ' <et>'
                    rows_list.append(row)
                elif row_length == ROW_LENGTH:
                    rows_list.append(row)

                for row_to_append in rows_list:
                    row_to_append['text'] = str2vec(row_to_append['text'], w2v)
                    # This if taking a lot of time because of check !?
                    if len(row_to_append['text']) != 0:
                        del row_to_append['author']
                        del row_to_append['time']
                        dict_list.append(row_to_append)
                        row = dict()
                        counter = counter + 1
                        # Wo wala counter
                        print(str(counter))
                    else:
                        for i in range(500):
                            row_to_append['text'].append(0)
                        del row_to_append['author']
                        del row_to_append['time']
                        dict_list.append(row_to_append)
                        row = dict()
                        counter = counter + 1
                        # Wo wala counter
                        print(str(counter))
        rows_list = []
    print('Loaded all data')
    df = pd.DataFrame(dict_list)
    df = df[df['text'].notna()]
    df = df[df['ispredator'].notna()]

    X = df['text'].to_numpy()
    X = np.array(list(x for x in X))
    y = df['ispredator'].to_numpy()

    print('About to check for labels')
    c = 1
    for ele in y:
        if ele == 1:
            print("1 counter = ", c)
            c = c + 1
    return X, y
