from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm 
import warnings
import pandas as pd 
from captor.loader import Loader
from captor.utils import get_batch



class SirGN:    
    def __init__(self,n1,n2,levels=10): 
        self.n=n1 # number of clusters
        self.n1=n1 # rep of feat of edge
        self.n2=n2 # final temporal representation
        self.levels=levels
        self.levelProcess=[(MinMaxScaler(),MiniBatchKMeans(n_clusters=self.n, random_state=1)) for i in range(levels)]
        self.edgeFeatures=(MinMaxScaler(),MiniBatchKMeans(n_clusters=self.n1, random_state=1))
        self.TemporalRepresentation=(MinMaxScaler(),MiniBatchKMeans(n_clusters=self.n2, random_state=1))
        
        
    def mapGraph(self,G,fe):
        dic={}
        invdic={}
        dicedge={}
        count=0
        count1=0
        lis=[]
        for v in G:
            if v not in dic:
                dic[v]=count
                invdic[count]=v
                count+=1
        for s in G:
            dicedge[s]={}
            for h in G[s]:
                # Adding extra layer of loop to get features from loader
                for t in G[s][h]:
                    dicedge[s][t]=count1
                    count1+=1
                    lis.append(G[s][h][t].reshape((1,G[s][h][t].shape[0])))
                
        return dic,invdic,dicedge,np.vstack(lis)

    
    def normalizeRow(self,emb1):
        er=0.000000001
        M=emb1.max(axis=1)
        m=emb1.min(axis=1)
        vec=np.abs(M-m)
        m2=m[vec>=er]
        m1=m2.reshape(m2.shape[0],1)
        emb=np.ones(emb1.shape)
        emb[vec>=er,:]=(emb1[vec>=er,:]-m1)/(M[vec>=er].reshape(m2.shape[0],1)-m1)
        emb[vec<er,:]=1/emb1.shape[1]
        #print(subx.shape)
        su=emb.sum(axis=1)
        #print(su.shape)
        emb2=np.ones(emb1.shape)
        vec1=np.abs(su)
        vec2=su[vec1>=er]
        emb2[vec1>=er,:]=emb[vec1>=er,:]/vec2.reshape(vec2.shape[0],1)
        emb2[vec1<er,:]=1/emb1.shape[1]
        return emb2

    
    def partial_fit_singlet(self,G,fe,fn):
        dic,invdic,dictedge,featuresedge=self.mapGraph(G,fe)
        n=self.n
        nv=len(dic)
        #edge Emb
        self.edgeFeatures[0].partial_fit(featuresedge)
        self.edgeFeatures[1].partial_fit(self.edgeFeatures[0].transform(featuresedge))
        embFeatures=self.edgeFeatures[1].transform(self.edgeFeatures[0].transform(featuresedge))
        embFeatures=self.normalizeRow(embFeatures)
        #initial Emb
        embin=(np.ones((nv,n))/n)*np.array([[len(G[invdic[x]]['in'])] for x in range(nv)])
        embout=(np.ones((nv,n))/n)*np.array([[len(G[invdic[x]]['out'])] for x in range(nv)])
        embfe=np.vstack([fn[invdic[x]].reshape((1,fn[invdic[x]].shape[0])) for x in range(nv)])
        emb=np.hstack([embfe,embin,embout])
        #loop
        for i in range(self.levels):
            (scaler,kmeans)=self.levelProcess[i]
            scaler.partial_fit(emb)
            kmeans.partial_fit(scaler.transform(emb))
            emb0=kmeans.transform(scaler.transform(emb))
            emb0=self.normalizeRow(emb0)
            #outedge
            embout=np.zeros((nv,n*self.n1))
            for v in G:
                for u in G[v]['out']:
                    hh=embFeatures[dictedge[v][u]]
                    hh=hh.reshape((hh.shape[0],1))
                    yy=emb0[dic[u]]
                    yy=yy.reshape((1,yy.shape[0]))
                    embout[dic[v]]+=np.matmul(hh,yy).flatten()
            #inedge
            embin=np.zeros((nv,n*self.n1))
            for v in G:
                for u in G[v]['in']:
                    hh=embFeatures[dictedge[u][v]]
                    hh=hh.reshape((hh.shape[0],1))
                    yy=emb0[dic[u]]
                    yy=yy.reshape((1,yy.shape[0]))
                    embin[dic[v]]+=np.matmul(hh,yy).flatten()
            emb=np.hstack([embfe,embin,embout]) 
        #temporal aggregation transformation
        (scaler,kmeans)= self.TemporalRepresentation
        scaler.partial_fit(emb)
        kmeans.partial_fit(scaler.transform(emb))
    
        
    def transform_singlet(self,G,fe,fn):
        dic,invdic,dictedge,featuresedge=self.mapGraph(G,fe)
        n=self.n
        nv=len(dic)
        #edge Emb
        embFeatures=self.edgeFeatures[1].transform(self.edgeFeatures[0].transform(featuresedge))
        embFeatures=self.normalizeRow(embFeatures)
        #initial Emb
        embin=(np.ones((nv,n))/n)*np.array([[len(G[invdic[x]]['in'])] for x in range(nv)])
        embout=(np.ones((nv,n))/n)*np.array([[len(G[invdic[x]]['out'])] for x in range(nv)])
        embfe=np.vstack([fn[invdic[x]].reshape((1,fn[invdic[x]].shape[0])) for x in range(nv)])
        emb=np.hstack([embfe,embin,embout])
        #loop
        for i in range(self.levels):
            (scaler,kmeans)=self.levelProcess[i]
            emb0=kmeans.transform(scaler.transform(emb))
            emb0=self.normalizeRow(emb0) #-> transition probability?
            #outedge
            embout=np.zeros((nv,n*self.n1))
            
            # parallaize 
            for v in G:
                for u in G[v]['out']:
                    hh=embFeatures[dictedge[v][u]]
                    hh=hh.reshape((hh.shape[0],1))
                    yy=emb0[dic[u]]
                    yy=yy.reshape((1,yy.shape[0]))
                    embout[dic[v]]+=np.matmul(hh,yy).flatten()
            #inedge
            embin=np.zeros((nv,n*self.n1))
            # parallaize 
            for v in G:
                for u in G[v]['in']:
                    hh=embFeatures[dictedge[u][v]]
                    hh=hh.reshape((hh.shape[0],1))
                    yy=emb0[dic[u]]
                    yy=yy.reshape((1,yy.shape[0]))
                    embin[dic[v]]+=np.matmul(hh,yy).flatten()
            #finalEmb
            emb=np.hstack([embfe,embin,embout]) 
        #temporal aggregation transformation
        (scaler,kmeans)= self.TemporalRepresentation
        emb0=kmeans.transform(scaler.transform(emb))
        emb=self.normalizeRow(emb0)
        mapRes={u:emb[dic[u]] for u in G}
        return mapRes
    
    def temporalUpdate(self,t,p,delta=1,alpha=1,current={}):        

        A_prev = current['A']
        M_prev = current["M"]
        F_prev = current['F']
        t_prev = current['T']
        
        p = p.reshape(-1,1)
                
        t_prev_j = t_prev/alpha
        t_j = t/alpha
        
        M_curr = np.exp(-(t_j - t_prev_j)/delta - (t_j - t_prev_j)) * M_prev + p
        A_curr = np.exp(-(t_j - t_prev_j)/delta ) * A_prev + M_curr @ p.reshape(-1,1).T
        
        F_curr = np.exp(-(t_j - t_prev_j)/delta ) * F_prev + p

        # Create a dict that has A_prev, M_prev, F_prev, T_prev
        current = {"A":A_curr,"M":M_curr,"F":F_curr,"T":t}
        return current
    
        
    def train(self,train_windows,node_feat,batch_size=5):
        start_point = 0
        
        # Batch training of the static sirgn
        for train_batch in tqdm(get_batch(train_windows, batch_size)):
            l=Loader()
            start_point = l.readUnion(train_batch,node_feat,start_point)
            self.partial_fit_singlet(l.G, l.edgeFeatures,l.nodeFeatures)
            
    
    def inference(self,data,node_feat,t,all_embd,INITIAL_VALUES,delta=1,alpha=1):
        
        # Inference for all graphs
        l = Loader()
        l.read_single_graph(data,node_feat) 
        
        # get the real node id from the loader           
        p = self.transform_singlet(l.G,l.edgeFeatures,l.nodeFeatures)
        # all_p.update({t:p})
        
        current_window_node_emb = {}
        for u in p.keys():
            u_real = l.revco[u]
            current = all_embd.get(u_real,INITIAL_VALUES)
            temp_embd = self.temporalUpdate(t,p[u],delta=delta,alpha=alpha,current = current)
            all_embd.update({u_real:temp_embd})
            current_window_node_emb.update({u_real:temp_embd})
            
        
        return all_embd, current_window_node_emb