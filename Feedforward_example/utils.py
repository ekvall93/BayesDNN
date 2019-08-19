import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
#from semiSupervised import *
import string

def get_train_data(df, clusters, model, s):
    num_datapoints = 0
    for i, nlf in enumerate(clusters):
        labels = df[str(nlf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        num_datapoints += len(labels)
    
        
    
    train_lab = list()
    
    np.random.seed(s)
    np.random.shuffle(clusters)
    
    for k, clf in enumerate(clusters):
        labels = df[str(clf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        vectors = model[labels]        
        if k==0:
            train_v = vectors
        else:
            train_v = np.concatenate((train_v, vectors))
             
        train_lab += labels

        
        
    return train_v, train_lab

def get_cluster_article_vectors(df, model, article_nr):
    labels = df[str(article_nr)].dropna().tolist()
    L = [str(int(l)) for l in labels if RepresentsInt(str(l)[0])]
    X = model[L]
    return X, L

def getVectors(df,model, potential_lf):
    for i, plf in enumerate(potential_lf):
        vec, labels = get_cluster_article_vectors(df,model,plf)
        if i ==0:
            X = vec
            L = np.asarray(labels)
        else:
            X = np.concatenate((X,vec))
            L = np.concatenate((L, labels))
    return X, L

def run_both_models(df, model, X_un_pot, topk, LF, NLF, combLF, combNLF):

    X_train, y_train, X_test, y_test = get_array_data(df, model, LF, NLF, [combLF[0]], [combNLF[0][0]])   

    print("--------------Vanilla------------------")

    sel_train = selfTrainer(topk=topk, num_iter=1000, vanillaSemiSupervised=True, save_error=True, verbose=True, equal_split=True)
    lfix1, nlfix1, miss_pred_lf1, miss_pred_nlf1, vanilla_val_error_list, vanilla_prob, vanilla_best_score = sel_train.fit(X_train, y_train, None, None, X_un_pot,X_test, y_test)

    X_train, y_train, X_test, y_test, X_val, y_val = get_array_data(df, model, LF, NLF, list(combLF), list(combNLF[0]))

    print("--------------New------------------")

    sel_train = selfTrainer(topk=topk, num_iter=1000, vanillaSemiSupervised=False, save_error=True, verbose=True, equal_split=True)
    lfix1, nlfix1, miss_pred_lf1, miss_pred_nlf1, non_vanilla_val_error_list , non_vanilla_test_error_list, not_vanilla_prob, not_vanilla_best_score = sel_train.fit(X_train, y_train, X_val, y_val, X_un_pot, X_test, y_test)

    return [vanilla_val_error_list, non_vanilla_val_error_list, non_vanilla_test_error_list, vanilla_prob, not_vanilla_prob, vanilla_best_score, not_vanilla_best_score,[combLF[0]], [combNLF[0][0]]]

def see_abstract(cluster, article_nr, df, df_abs):
    ix = df[str(cluster)][article_nr]
    print(df_abs[df_abs.Doc_id == int(ix)].Abstracts.values)

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_nr_cluster(cluster, df):
    nr_article = 0
    article_id = list()
    for v in df[str(cluster)].dropna():
        v = str(v)
        if not v.islower():
            nr_article +=1
            article_id.append(v)
    return nr_article, article_id

def get_ix(nr_total_article, nr_article=10, seed=7):
    ix = np.arange(nr_total_article)
    np.random.seed(seed)
    np.random.shuffle(ix)
    return ix[:nr_article]
def sample_abstrac(cluster_nr, df,df_abs, max_nr_article=10):
    nr_article, article_id = get_nr_cluster(cluster_nr, df)
    all_ix = get_ix(nr_article, max_nr_article)
    for i, (ix,ixx) in enumerate(zip(all_ix, article_id)):
        print("------------",i,"   Article number:",ixx)
        see_abstract(cluster_nr, ix, df, df_abs)

def concatenate_and_get_labels(LF, NLV):
    X = np.concatenate((LF, NLV))
    y = np.concatenate((np.zeros(LF.shape[0]), np.ones(NLV.shape[0])))
    return X, y

def get_labels_and_vec(df, clusters, s):
    num_datapoints = list()
    for i, nlf in enumerate(clusters):
        labels = df[str(nlf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        num_datapoints.append(len(labels))

    print(np.around(np.array(num_datapoints) / sum(num_datapoints) *100))
    nr_test_points = int(sum(num_datapoints) * 0.1)

    test_lab = list()
    train_lab = list()
    np.random.seed(s)
    np.random.shuffle(clusters)
    train_phase = False
    i = 0

    while len(test_lab) < nr_test_points:
        labels = df[str(clusters[i])].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        vectors = model[labels]
        if i==0:
            test_v = vectors
        else:
            test_v = np.concatenate((test_v, vectors))
        test_lab += labels
        i +=1


    for k, clf in enumerate(clusters[i:]):
        labels = df[str(clf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        vectors = model[labels]
        if k==0:
            train_v = vectors
        else:
            train_v = np.concatenate((train_v, vectors))

        train_lab += labels

    return train_v, train_lab, test_v, test_lab

def get_distibutions(df, clusters):
    num_datapoints = list()
    for i, nlf in enumerate(clusters):
        labels = df[str(nlf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        num_datapoints.append(len(labels))

    return np.around(np.array(num_datapoints) / sum(num_datapoints) *100)

def get_data(df, model, clusters, test_val_clsuter):
    clusters = list(set(clusters) - set(test_val_clsuter))
    test_lab = list()
    train_lab = list()
    val_lab = list()

    for i in test_val_clsuter:
        labels = df[str(i)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        vectors = model[labels]
        if i in test_val_clsuter:
            if i == test_val_clsuter[0]:
                test_v = vectors
                test_lab += labels
            else:
                val_v = vectors
                val_lab += labels


    for k, clf in enumerate(clusters):
        labels = df[str(clf)].dropna().apply(lambda x: str(np.int(x)) if isinstance(x, np.int) or isinstance(x, np.float) else x).tolist()
        vectors = model[labels]
        if k==0:
            train_v = vectors
        else:
            train_v = np.concatenate((train_v, vectors))

        train_lab += labels

    if len(test_val_clsuter) > 1:
        return train_v, train_lab, test_v, test_lab, val_v, val_lab
    else:
        return train_v, train_lab, test_v, test_lab


def get_array_data(df,model, LF,NLF, LF_holdout_cluster, NLF_holdout_cluster):
    if len(LF_holdout_cluster) == 1:
        LF_train_v, LF_train_lab, LF_test_v, LF_test_lab = get_data(df, model, LF, LF_holdout_cluster)
        NLF_train_v, NLF_train_lab, NLF_test_v, NLF_test_lab = get_data(df, model, NLF, NLF_holdout_cluster)
    else:
        LF_train_v, LF_train_lab, LF_test_v, LF_test_lab, LF_val_v, LF_val_lab = get_data(df, model, LF, LF_holdout_cluster)
        NLF_train_v, NLF_train_lab, NLF_test_v, NLF_test_lab, NLF_val_v, NLF_val_lab = get_data(df, model, NLF, NLF_holdout_cluster)

    X_train, y_train = concatenate_and_get_labels(LF_train_v, NLF_train_v)
    X_test, y_test = concatenate_and_get_labels(LF_test_v, NLF_test_v)

    if len(LF_holdout_cluster) == 1:
        return X_train, y_train, X_test, y_test
    else:
        X_val, y_val = concatenate_and_get_labels(LF_val_v, NLF_val_v)
        return X_train, y_train, X_test, y_test, X_val, y_val


def create_plt(kth_dep, to_series=True):
    if to_series:
        kth_dep_s = pd.Series(kth_dep).value_counts().sort_values(ascending=False)
    else:
        kth_dep_s = kth_dep
    
    
    plt.figure(figsize=(40,20))
    ax = sns.barplot(kth_dep_s.index, kth_dep_s.values)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right",fontsize=28)

    ax.set_ylabel("Counts",fontsize=60)

    ax.set_title("KTH authors distribution(departments)",fontsize=80)




    sns.set(font_scale=3)
    plt.gcf().subplots_adjust(bottom=0.45)

    plt.show()


def cluster_article_dep(df_cluster, df_abs, df_auth):
    df_cluster_authors = pd.DataFrame()
    for i, col in enumerate(df_cluster.columns):
        id_list = list()
        for row in df_cluster[col]:
            if row is np.nan:
                break
            else:
                if not str(row).islower():
                    auth_id = df_abs[df_abs.Doc_id == int(row)].KTH_id.tolist()[0].split(":")
                    kth_dep = list()
                    for a in auth_id:
                        if len(df_auth[df_auth.KTH_id == a].dep_name.values) > 0:
                            kth_dep += df_auth[df_auth.KTH_id == a].dep_name.unique().tolist()
              
                    id_list.append(max(set(kth_dep), key=kth_dep.count))
                else:
                    break
        #LIST.append(id_list)
        df_new = pd.DataFrame({int(col): id_list})
        df_cluster_authors = pd.concat([df_cluster_authors, df_new], axis=1)
    return df_cluster_authors

def authors_dep_and_school(df, author_names):
    kth_dep = list()
    kth_school = list()
    no_dep=0
    for name in author_names:
        i = name
        if len(df[df.KTH_id == i].dep_name.values) > 0:
            kth_dep += df[df.KTH_id == i].dep_name.unique().tolist()
            kth_school += df[df.KTH_id == i].skola_namn.unique().tolist()
        else:
            no_dep +=1
    return kth_dep, kth_school

def get_lists_of_dep_and_school(df_auth, positive_author_list):
    kth_dep_list, kth_school_list = list(), list()
    for positive_author_ in positive_author_list:
        kth_dep, kth_school = authors_dep_and_school(df_auth, positive_author_)
        kth_dep_list.append(kth_dep) 
        kth_school_list.append(kth_school)
    return kth_dep_list, kth_school_list


def get_central_words(x, model, topn=20):
    x = x.dropna()
    words = list()
    for e in x:
        if str(e)[0].islower():
            words.append(e)
    words = np.asarray(words)
    val = cosine_similarity(model[words])
    sumVal = val.sum(axis=0)
    ix = np.array(np.argsort(sumVal))[::-1]
    return words[ix[:topn]]


class get_author_lits(object):
    def get_words(self, arr):
        words = list()
        for v in arr:
            if not v[0].isupper():
                words.append(v)
        return words
    def get_author(self, arr):
        author = list()
        for v in arr:
            if v in list(id_to_auth.keys()):
                author.append(v)
        return author
    def get_author_lits(self, cluster_list, df):
        author_list = list()
        for cluster in cluster_list:
            author_list.append(self.get_author(df[int(cluster)].dropna().values))
        return author_list

def cluster_mapping(df_auth, postive_clusters, auth_to_id):
    cluster_dict = dict()
    for lists in postive_clusters:
        for tuples in lists:
            cluster = tuples[0]
            name = tuples[1]
            dep = df_auth[df_auth.KTH_id == auth_to_id[name]].dep_name.unique().tolist()[0]

            if not cluster in cluster_dict.keys():
                cluster_dict[cluster] = name + "("+str(dep)+")"
            else:
                cluster_dict[cluster] += "; "+ name + "(" + str(dep) + ")"
    return cluster_dict


def df_convert_name_to_ids(df, auth_to_id):
    l = df.shape[1]
    for i in list(df.columns):
        L = df[str(i)].shape[0]
        for k in range(L):
            entry = df[str(i)][k]
            if isinstance(entry, str):
                if entry[0].isupper():
                    df[str(i)][k] = auth_to_id[str(entry)]
                else:
                    break
            else:
                break
    return df


def remove_cluster_withoutAuthors(df):
    l = df.shape[1]
    for i in range(l):
        if str(df[str(i)][0]) == "nan":
            df.drop(columns=[str(i)], inplace=True)
        else:
            if str(df[str(i)][0])[0].islower():
                df.drop(columns=[str(i)], inplace=True)
    return df

def get_author_in_cluster(df_cluster, df_authors):
    invalidChars = set(string.punctuation.replace("_", ""))
    df_cluster_authors = pd.DataFrame()
    for i, col in enumerate(df_cluster.columns):
        id_list = list()
        for row in df_cluster[col]:
            if row is np.nan:
                break
            else:
                if not str(row).islower():
                    try:
                        id_list += df_authors[df_authors.Doc_id == int(row)].KTH_id.tolist()
                    except:
                        print(row)
                else:
                    break
        df_new = pd.DataFrame({int(col): id_list})
        df_cluster_authors = pd.concat([df_cluster_authors, df_new], axis=1)
    return df_cluster_authors

def convert_lists_2_df(lists):
    df = pd.DataFrame()
    for i, x in enumerate(lists):
        df_new = pd.DataFrame({'cluster' + str(i):x})
        df = pd.concat([df, df_new], ignore_index=True, axis=1)
    return df

def extract_article_vectors(doc_tag, doc_vec, id_to_auth):
    auth_tag = list()
    auth_vec = list()
    for i, da in enumerate(doc_tag):
        if da[0] != "u":
            auth_vec.append(doc_vec[i])
            auth_tag.append(da)
    return auth_vec, auth_tag

def get_cluster_containing_author(dataFrame, S):
    mask = np.column_stack([dataFrame[col].astype(str).str.contains(str(S), na=False) for col in dataFrame])
    cluster = list()
    for i, col in enumerate(dataFrame):
        if dataFrame[col].astype(str).str.contains(str(S), na=False).any():
            cluster.append((col, id_to_auth[str(S)]))
    return cluster

def get_cluster(f, df, ids):
    pool = Pool(processes=(cpu_count() - 1))

    func = partial(f, df)
    result = pool.map_async(func, ids)
    return result.get()


def show_abstrcts(name, df):
    print(df[df.KTH_name.str.contains(name)].values)



def get_related_authors(name, model, auth_to_id, id_to_auth, topn=50):    
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """  
    author_Series = pd.Series(list(auth_to_id.keys()))
    if not any(author_Series == name):
        check_name = author_Series.str.contains(str(name))
        if any(check_name):
            print("Did you mean any ofese name/s:")
            for a in author_Series[check_name].unique(): 
                print(a)
        else:
            print("Couldn't find any good mathc")
    else:
        tag = auth_to_id[name]
        
        for r in model.docvecs.most_similar(tag,topn=topn):
            if r[0][0] == "u":
                print(id_to_auth[r[0]],r[1])

def get_related_words(name, model, auth_to_id, id_to_auth, topn=50):
    
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """  
    author_Series = pd.Series(list(auth_to_id.keys()))
    if not any(author_Series == name):
        check_name = author_Series.str.contains(str(name))
        if any(check_name):
            print("Did you mean any ofese name/s:")
            for a in author_Series[check_name].unique(): 
                print(a)
        else:
            print("Couldn't find any good mathc")
    else:
        tag = auth_to_id[name]
        tag_v = model[str(tag)]
        for r in model.most_similar([tag_v], topn=topn):
            print(r)

def semantic_search_author(sentence, model, df, topn=30):
    word_2_vec = 0
    for word in sentence:
        word_2_vec += model[str(word)]
    for a in model.docvecs.most_similar( [ word_2_vec ], topn=topn):
        if a[0][0] !="u":
            print(str(a[0]),"||",get_article_authors_name(int(a[0]),df)," || ",np.around(a[1],2))

def semantic_search_word(sentence, model, df, topn=30):
    word_2_vec = 0
    for word in sentence:
        word_2_vec += model[str(word)]
    for a in model.most_similar( [ word_2_vec ], topn=topn):
        if a[0] not in sentence:
            print(a)

def get_article_authors_name(doc_id, df):
    name = ""
    if len(df[df.Doc_id == doc_id].KTH_name) > 0:
        name_list = df[df.Doc_id == doc_id].KTH_name.values[0].split(":")
        for i, n in enumerate(name_list):
            if i == 0:
                name += str(n)
            elif i == len(name_list) - 1 and i > 1:
                name += " and " + str(n)
            elif i > 1:
                name += " , " + str(n)
    else:
        name = "NaN"
        
    return name

class pickle_obj:
    def save(self, obj, name ):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    def load(self, name ):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

def cosine_sim_mat(X,return_sim = False):
    dist = X.T.dot(X)
    norm = (X * X).sum(0, keepdims=True) ** .5
    doc_sim = dist / (norm * norm.T)
    doc_dist = 2 - dist / norm / norm.T
    if return_sim:
        return doc_dist, doc_sim
    else:
        return doc_dist

def save_classifer(model, path):
    # save the classifier
    with open(path + '.pkl', 'wb') as fid:
        pickle.dump(model, fid)

def load_classifier(path):        
    # load it again
    with open(path + '.pkl', 'rb') as fid:
        BGMM_loaded = pickle.load(fid)
    return BGMM_loaded

translate_dict = dict();
translate_dict["ö"] = "o"
translate_dict["ä"] = "a"
translate_dict["å"] = "a"
translate_dict["Ö"] = "O"
translate_dict["Ä"] = "A"
translate_dict["Å"] = "A"
translate_dict["é"] = "e"
translate_dict["ü"] = "u"
translate_dict["á"] = "a"
translate_dict["Á"] = "A"
translate_dict["\xad"] = ""
translate_dict["æ"] = "a"
translate_dict["¡"] = "i"
translate_dict["«"] = "a"
translate_dict["ó"] = "O"
translate_dict["œ"] = "o"
translate_dict["\xa0"] = ""
translate_dict["ç"] = "c"
translate_dict["ñ"] = "n"
translate_dict["ú"] = "u"
translate_dict[","] = ""
translate_dict["è"] = "e"
translate_dict["‰"] = ""
translate_dict["‰"] = ""
translate_dict["£"] = "E"
translate_dict["“"] = ""
translate_dict["ˆ"] = ""
translate_dict["\x8d"] = ""
translate_dict["™"] = ""
translate_dict["²"] = ""
translate_dict["Ì"] = "I"
translate_dict["\x81"] = ""
translate_dict["Œ"] = "E"
translate_dict["´"] = ""
translate_dict["¸"] = ""
translate_dict["…"] = ""
translate_dict["ø"] = "o"


def make_name_noAscii(ascii_name):
    name = str()
    for l in ascii_name:
        if ord(l) < 128:
            name +=l
        else:
            name += translate_dict[l]
    return name


from multiprocessing import Pool
from functools import partial

import time
def fk(values, keys, C):
    # For the first 10 clusters
    for cluster in C:
        #words = []
        #for i in range(0,len(wc.values())):
        #    if( list(wc.values())[i] == cluster ):
        #        words.append(list(wc.keys())[i])
        #return words
        return list(keys[values==cluster])

def get_cluster_members(num_clusters, word_centroid_map):
    chunks = [[i] for i in range(0, num_clusters)]
    pool = Pool(processes=cpu_count() - 1)
    values = np.array(list(word_centroid_map.values()))
    keys = np.array(list(word_centroid_map.keys()))
    func = partial(fk, values, keys)
    result = pool.map_async(func, chunks)
    return result.get()


def get_cluster_containing_substring(S,dataFrame):
    mask = np.column_stack([dataFrame[col].astype(str).str.contains(str(S), na=False) for col in dataFrame])
    cluster = list()
    for i, col in enumerate(dataFrame):
        if dataFrame[col].astype(str).str.contains(str(S), na=False).any():
            cluster.append(col)
    return cluster
def aggregating_seperate(clsuters, df):
    aggregated_ids = list()
    for p in clsuters:
        p_sampes = np.asarray(df[str(p)].values).astype(str)
        p_sampes = p_sampes[~(p_sampes == "nan")]
        aggregated_ids.append(list(p_sampes))
    return aggregated_ids

def prepare_data_for_model(tags, tag_to_prob, map_to_y=0):
    prob_vec = list()
    for t in tags:
        prob_vec.append(tag_to_prob[t])
    if map_to_y == 0:
        y = np.zeros((len(prob_vec),1))
    elif map_to_y is None:
        return np.asarray(prob_vec)
    else:
        y = np.ones((len(prob_vec),1)) * map_to_y
    return y, np.asarray(prob_vec)

def split(X,y,ix):
    for e, i in enumerate(ix):
        if e == 0:
            Xs = X[i]
            ys = y[i]
        else:
            Xs = np.concatenate((Xs, X[i]))
            ys = np.concatenate((ys, y[i]))
    return Xs, ys

def split_data(X,y,ratio):
    n = len(X)
    ix = np.arange(n)
    np.random.shuffle(ix)
    train_ix = ix[int(n*ratio):]
    test_ix = ix[:int(n*ratio)]
    X_train, y_train = split(X,y,train_ix)
    X_test, y_test = split(X,y,test_ix)
    return X_train, X_test, y_train, y_test

#pickle_o = pickle_obj(); 
#id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
#auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")
