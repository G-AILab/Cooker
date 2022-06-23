import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import he_uniform, glorot_uniform, lecun_uniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import plot_model, multi_gpu_model, model_to_dot
from IPython.display import SVG
from keras_tqdm import TQDMNotebookCallback, TQDMCallback
from sklearn.manifold import TSNE
from tqdm import tqdm, tqdm_pandas, tqdm_notebook, tqdm_gui
from sklearn.preprocessing import *

class RandomWalker(object):
    def __init__(self,nxG):
        super(RandomWalker, self).__init__()
        self.G = nxG

    def _walk(self, start_node, length_walk):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < length_walk:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand()*len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk

    def _simulate_walks(self, length_walk, num_walks):
        # Repeatedly simulate random walks from each node.
        walks = []
        nodes = list(self.G.nodes())
        for walk_iter in (range(num_walks)):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, length_walk))
        return walks
    
def transform_neighbor_korder_shuffle(x, k):
    return random.choices(x, k=k)

def transform_degree(x):
    return list(dict(G.degree(x)).values)

# def transform_self_korder_deepwalk(x, k):
#     return rd._walk(x, k)

def get_transform(f,n,a,k):
    return a[f(n,k)]

class AggEmbedding(object):
    def __init__(self,G,G_attr,G_label,uv_pairs,task):
        super(AggEmbedding,self).__init__()
        self.G = G
        self.G_label = G_label
        self.uv_pairs = uv_pairs
        self.task = task
        
        if G_attr.shape[1] >= 2000:
            nodes_matrix = G_attr.drop('nodes', axis=1).values
            M = sp.csr_matrix(nodes_matrix)
            U, S, V = sp.linalg.svds(M.asfptype(), 128)
            W = U * S**0.5
            self.attribute = W / np.linalg.norm(W, axis=1, keepdims=True)
        else:
            W = G_attr.drop('nodes', axis=1).fillna(0).values
            self.attribute = W / np.linalg.norm(W, axis=1, keepdims=True)
            
    def MultiTask_SelfsSupervision_AE(self,ushape,vshape):
    
        '''
        input_1、input_2：u、v
        shape: (479160, 1433)
        '''
        input_1 = Input(shape=(ushape,))
        output_1 = Dense(512)(input_1)


        input_2 = Input(shape=(vshape,))
        output_2 = Dense(512)(input_2)


        conv = concatenate([
            input_1,input_2,
            output_1,output_2,
        ])

        conv = BatchNormalization()(conv)
        conv = Dense(128, name='pair_embedding')(conv) 

        label1 = Dense(ushape,activation='softmax',name='loss1')(conv)
        label2 = Dense(vshape,activation='softmax',name='loss2')(conv)
        label3 = Dense(1,activation='sigmoid',name='loss3')(conv)
        label4 = Dense(1,activation='sigmoid',name='loss4')(conv)

        model = Model(inputs=[input_1,input_2], outputs=[label1,label2,label3,label4])
        return model
    
    def model(self,data,lr):
        K.clear_session()
        
        u_attr = np.concatenate(data['e1'].values).reshape(data.shape[0],-1)
        v_attr = np.concatenate(data['e2'].values).reshape(data.shape[0],-1)
        
        model = self.MultiTask_SelfsSupervision_AE(u_attr.shape[1],v_attr.shape[1])
        model.compile(optimizer=Adam(lr=lr),
              loss={
                  'loss1' : 'kullback_leibler_divergence',
                  'loss2' : 'kullback_leibler_divergence',
                  'loss3' : 'binary_crossentropy',
                  'loss4' : 'binary_crossentropy',
              },
              metrics={
                  'loss1' : 'kullback_leibler_divergence',
                  'loss2' : 'kullback_leibler_divergence',
                  'loss3' : 'acc',
                  'loss4' : 'acc',
              })
        
        tf_input = [
            StandardScaler().fit_transform(u_attr),
            StandardScaler().fit_transform(v_attr),
        ]
        tf_output = [
            u_attr,
            v_attr,
            data['uv'].values,
            data['agg'].values,
        ]
        callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)]
        model.fit(tf_input,
                  tf_output,
                  epochs=10,
                  batch_size=1024,
                  shuffle=True,
                  verbose=1,
                  callbacks=callbacks)

        embed = Model(inputs=model.input,
                      outputs=model.get_layer('pair_embedding').output)
        embedding = embed.predict(tf_input, batch_size=2048, verbose=0)
        return embedding,model
    
    def generate_data(self,G,df):
        if self.task == 'link':
            mask_G = nx.Graph()
            mask_G.add_nodes_from(G.nodes())
            mask_G.add_edges_from(df.values)
        if self.task == 'node':
            mask_G = G

        rd = RandomWalker(mask_G)

        feature = []
        k = 0
        i=0
        for u,v in tqdm_notebook(df.values):
            col = []
            if u==v:
                k = 1
            else:
                k = 0
            for t in [3,5,10]:

                d1,d2 = get_transform(rd._walk,u,self.attribute,t), get_transform(rd._walk,v,self.attribute,t) 
                feature.append([u,v,np.max(d1,axis=0), np.max(d2,axis=0),k,1])
                feature.append([u,v,np.mean(d1,axis=0), np.mean(d2,axis=0),k,1])
                feature.append([u,v,np.min(d1,axis=0), np.min(d2,axis=0),k,1])
                feature.append([u,v,np.sum(d1,axis=0), np.sum(d2,axis=0),k,1])

                feature.append([u,v,np.sum(d1,axis=0), np.min(d2,axis=0),k,0])
                feature.append([u,v,np.sum(d1,axis=0), np.max(d2,axis=0),k,0])
                feature.append([u,v,np.max(d1,axis=0), np.min(d2,axis=0),k,0])
                feature.append([u,v,np.mean(d1,axis=0), np.max(d2,axis=0),k,0])
                feature.append([u,v,np.mean(d1,axis=0), np.sum(d2,axis=0),k,0])
                feature.append([u,v,np.mean(d1,axis=0), np.min(d2,axis=0),k,0])
            i+=1
        data = pd.DataFrame(feature,columns=['u','v','e1','e2','uv','agg'])
        return data
    
    def fit(self):
        lr=0.001
        data = self.generate_data(self.G,self.uv_pairs)
        
        embedding,agg = self.model(data,lr)
        
        jb = data[['u','v']]
        pairs_embedding = pd.DataFrame(
                    embedding, columns=["{}_pair".format(i) for i in range(embedding.shape[1])])
        pairs_col = list(pairs_embedding.columns)
        jb[pairs_col] = pairs_embedding

        edges_embedding = jb[['u','v'] + pairs_col]
        hu, hv = jb[['u'] + pairs_col].rename(
            columns={'u': 'nodes'}), jb[['v'] + pairs_col].rename(
                columns={'v': 'nodes'})
        edges = pd.concat([hu, hv], axis=0, ignore_index=True)
        edges = edges.groupby(['nodes'])[pairs_col].agg(['mean','sum']).sort_index().values
        edges = edges / np.linalg.norm(edges, axis=1, keepdims=True)
        self.nodes_embedding = pd.DataFrame(
            edges,
            columns=["embedding_{}".format(i) for i in range(edges.shape[1])])
        return self.nodes_embedding
