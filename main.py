import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import igraph as ig
import math
import random
import time
import os
from tqdm import tqdm, tqdm_pandas, tqdm_notebook, tqdm_gui
from sklearn.preprocessing import *

import scipy
from scipy import sparse as sp
from IPython.display import display
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import *
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegressionCV
import bhtsne

from utils import *
from model import *

%matplotlib inline

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
physical_devices = tf.config.list_physical_devices('GPU')

#task: node classification:'node'; link prediction: 'link'
task = 'node'
dataset = 'cora'
iG,G,G_label,G_attr = read_data(dataset)
num_classes = G_label['label'].map(lambda x:x[0]).nunique()
num_nodes = iG.vcount()

#model train
if task == 'node':
    df = pd.DataFrame(G.edges(),columns=['u','v']).sort_values(by=['u','v']).reset_index(drop=True)
    
    model = AggEmbedding(G,G_attr,G_label,df,task)
    nodes_embedding = model.fit()

    get_cv_score(nodes_embedding, G, G_label, [LogisticRegression(n_jobs=-1)], "{} {}".format(
                        'Pair Embedding',
                        "123"))
    
if task == 'link':
    mask_link_train,mask_link_test,label_train,label_test,df = get_train_test(G.edges(),num_nodes)
    
    model = AggEmbedding(G,G_attr,G_label,df,task)
    nodes_embedding = model.fit()
    
    nodes_first = nodes_embedding.iloc[mask_link_train.iloc[:,0],:].reset_index(drop=True)
    nodes_second = nodes_embedding.iloc[mask_link_train.iloc[:,1],:].reset_index(drop=True)
    pred_train = nodes_first*nodes_second
    pred_train = np.array(pred_train.sum(axis=1)).reshape(-1,1)

    nodes_first = nodes_embedding.iloc[mask_link_test.iloc[:,0],:].reset_index(drop=True)
    nodes_second = nodes_embedding.iloc[mask_link_test.iloc[:,1],:].reset_index(drop=True)
    pred_test = nodes_first*nodes_second
    pred_test = np.array(pred_test.sum(axis=1)).reshape(-1,1)

    clf = LogisticRegressionCV(Cs=10,max_iter=100,n_jobs=10,verbose=1,scoring='roc_auc') 
    clf.fit(pred_train,label_train)
    auc = roc_auc_score(label_test,clf.predict_proba(pred_test)[:,1])
    print("Validation SET ROC-AUC Score {} ".format(auc))