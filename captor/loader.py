import numpy as np

class Loader:
    def __init__(self):
        self.countID=0
        self.G={}
        self.co={}
        self.revco={}
        self.nodeFeatures=[]
        self.nodeFeatures_all = []
        self.edgeFeatures = {}
    
    def nodeID(self,x, f):
        if x not in self.co:
            self.co[x]=self.countID
            self.countID=self.countID+1
            node_id = x.split('_')[-1]
            self.revco[self.co[x]]=x
            self.nodeFeatures.append(f[node_id])
        return self.co[x]
    
    def read_single_graph(self,file,nodeFeaturesMap):
        x=file.values
        for a in range(x.shape[0]):
            i=self.nodeID(str(int(x[a,0])),nodeFeaturesMap)
            j=self.nodeID(str(int(x[a,1])),nodeFeaturesMap)
            self.addEdge((i,j),x[a,3:])
    
    def read(self,file,nodeFeaturesMap,l="",is_read_union=False):
        x=file.values
        for a in range(x.shape[0]):
            i=self.nodeID(str(l)+'_'+str(int(x[a,0])),nodeFeaturesMap)
            j=self.nodeID(str(l)+'_'+str(int(x[a,1])),nodeFeaturesMap)
            self.addEdge((i,j),x[a,3:])    
            
       
        if not is_read_union:
            self.nodeFeatures=np.vstack(self.nodeFeatures)
            self.edgeFeatures = np.vstack(self.edgeFeatures)

        
    def readUnion(self,graphs,all_nodefeats,start_point):
        
        for i,l in enumerate(graphs):
            self.read(l,all_nodefeats,str(i+start_point),is_read_union=True)
        
        self.nodeFeatures=np.vstack(self.nodeFeatures)
        
        return i

        
        
    def storeEmb(self,file,data):
        file1 = open(file, 'a') 
        for a in range(data.shape[0]):
            s=''+str(self.revco[a])
            for b in range(data.shape[1]):
                s+=' '+str(data[a,b])
            file1.write(s+"\n")
        file1.close()
            

    def addEdge(self,s,f):
        (l1,l2)=s
        if l1 not in self.G:
            self.G[l1]={}
            self.G[l1]['in']={}
            self.G[l1]['out']={}
        if l2 not in self.G:
            self.G[l2]={}
            self.G[l2]['in']={}
            self.G[l2]['out']={}
        self.G[l1]['out'][l2]=f
        self.G[l2]['in'][l1]=f
        edg_feat = self.edgeFeatures.get(l1)
        if edg_feat is None:
            self.edgeFeatures[l1] = {l2:f}

        else:
            edg_feat.update({l2:np.ones(1)})
            self.edgeFeatures[l1] = edg_feat
