import igraph as ig
import networkx as nx
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm, tqdm_pandas, tqdm_notebook, tqdm_gui
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

def NX_to_IG(G, directed=False):
    return ig.Graph(len(G),
                    list(zip(*list(zip(*nx.to_edgelist(G)))[:2])),
                    directed=directed)

def read_data(name):
    G = nx.read_adjlist("input/{}/{}_adjlist.txt".format(name, name),
                        delimiter=' ',
                        nodetype=int,
                        create_using=nx.DiGraph())
    #G.add_edges_from([i[::-1] for i in list(G.edges())])
    G_label = pd.read_pickle("input/{}/{}_label.pickle".format(name, name))
    G_attr = pd.read_pickle("input/{}/{}_attr.pickle".format(name, name))
    G_label['label'] = G_label['label'].map(lambda x: [x])

    iG = NX_to_IG(G, False)
    for i in tqdm_notebook(range(iG.vcount())):
        G.add_edge(i, i)

    print("{} Have {} Nodes, {} Edges, {} Attribute, {} Classes".format(
        name, iG.vcount(), iG.ecount(), G_attr.shape[1] - 1,
        G_label['label'].astype('str').nunique()))

    return iG, G, G_label, G_attr



def get_cv_score(emb, G, G_label, clf, name):
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    tot = 0
    for i in clf:  # svc_linear,svc_rbf,
        k, k1 = [], []
        print(i)
        for test_size in tqdm_notebook(ratios):
            train, test, train_label, test_label = train_test_split(
                emb,
                G_label['label'].map(lambda x: x[0]).values,
                test_size=1 - test_size)
            try:
                print('try:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            except:
                print('except:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            micro = "%0.4f±%0.4f" % (scores_clf['test_f1_micro'].mean(),
                                     scores_clf['test_f1_micro'].std() * 2)
            macro = "%0.4f±%0.4f" % (scores_clf['test_f1_macro'].mean(),
                                     scores_clf['test_f1_macro'].std() * 2)
            k.append([micro, macro])
            i.fit(train.astype(np.float32), train_label.astype(np.float32))
            k1.append([
                f1_score(test_label, i.predict(test.astype(np.float64)), average='micro'),
                f1_score(test_label, i.predict(test.astype(np.float64)), average='macro')
            ])

        tr = pd.DataFrame(k).T
        tr.columns = ['ratio {}'.format(i) for i in ratios]
        tr.index = ['train-micro', 'train-macro']

        display(tr)

    return tr


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges.iloc[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges.iloc[0,i]
            node2 = edges.iloc[1,i]
            if node1==node2 and node_count[node1]==2:
                index_train.append(i)

            elif node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break

            else:
                index_train.append(i)

        index_train = index_train + list(range(i + 1, e))
        index_test = index_val

        edges_train = edges.iloc[:, index_train]
        edges_test = edges.iloc[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        edges_train = edges.iloc[:,:split1]
        edges_test = edges.iloc[:,split1:]

    edges_train = edges_train.T
    edges_test = edges_test.T
    edges_train.columns = ['u','v']
    edges_test.columns = ['u','v']

    return edges_train.reset_index(drop=True), edges_test.reset_index(drop=True)

def get_edge_mask_link_negative(mask_link_positive,num_nodes,num_negative_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive.iloc[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = pd.DataFrame(np.zeros([2,num_negative_edges]))
    for i in range(num_negative_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative.iloc[:,i] = mask_temp
                break
    mask_link_negative = mask_link_negative.T
    mask_link_negative.columns = ['u','v']
    return mask_link_negative.reset_index(drop=True)


def get_train_test(edges,num_nodes,remove_ratio=0.2):
    mask_link_positive = pd.DataFrame(edges,columns=['u','v']).sort_values(by=['u','v']).reset_index(drop=True)
    mask_link_positive_train, mask_link_positive_test = split_edges(mask_link_positive.T, remove_ratio,True)

    mask_link_negative_train = get_edge_mask_link_negative(mask_link_positive,num_nodes,mask_link_positive_train.shape[0])
    mask_link_negative_test = get_edge_mask_link_negative(
        pd.concat([mask_link_positive,mask_link_negative_train],axis=0)
        ,num_nodes
        ,mask_link_positive_test.shape[0])

    mask_link_train = pd.concat([mask_link_positive_train,mask_link_negative_train],axis=0)
    mask_link_test = pd.concat([mask_link_positive_test,mask_link_negative_test],axis=0)

    label_positive_train = np.ones([mask_link_positive_train.shape[0],])
    label_negative_train = np.zeros([mask_link_positive_train.shape[0],])
    label_train = np.concatenate((label_positive_train,label_negative_train))

    label_positive_test = np.ones([mask_link_positive_test.shape[0],])
    label_negative_test = np.zeros([mask_link_positive_test.shape[0],])
    label_test = np.concatenate((label_positive_test,label_negative_test))

    df = pd.concat([mask_link_positive_train,mask_link_negative_train],axis=0)

    return mask_link_train,mask_link_test,label_train,label_test,df
