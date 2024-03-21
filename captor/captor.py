import numpy as np
from tqdm import tqdm
import pandas as pd
import json
from captor.sirgn import SirGN
from captor.loader import Loader
from captor.vae import VariationalAutoencoder
from captor.utils import get_frequency

class CAPTOR:
    """_summary_
    """
    def __init__(self,args,data_path,model_path):
        self.args = args
        self.root = data_path
        self.model_path = model_path
                
        
    def _get_data(self):
        print("Reading Data")
        self.train_data = pd.read_csv(f'{self.root}/{self.args.dataset}/train.csv')
        self.test_data = pd.read_csv(f'{self.root}/{self.args.dataset}/test.csv') 
        self.node_map = json.load(open(f"{self.root}/{self.args.dataset}/node_map.json"))
        
        
        self.label = self.test_data.label
        
        self.train_data = self.train_data[['src_computer','dst_computer','snapshot']]
        self.test_data = self.test_data[['src_computer','dst_computer','snapshot']]
    
        
        # Assigning 1 as node feature for all nodes
        # Other features can be used for future experiments
        self.node_feat_map = {}
        for u in self.node_map.values():
            self.node_feat_map.update({str(u):np.ones(1)})


        # Grouping data by snapshot to create graph
        self.train_data_group = self.train_data.groupby('snapshot') 
        
        # Getting snapshots
        self.train_t = list(self.train_data_group.groups.keys())
        self.train_windows = [get_frequency(self.train_data_group.get_group(i)) for i in self.train_data_group.groups.keys()]


        self.test_data_group = self.test_data.groupby("snapshot")
        self.test_t = list(self.test_data_group.groups.keys())
        
        
    def _train_sirgn(self):
        self._get_data()

        self.model  = SirGN(self.args.clustersize,self.args.dimensize,levels=5)
        self.model.train(self.train_windows, self.node_feat_map)


    
    def _get_train_node_embedding(self):
        
        emb_size = self.model.n2
        A_prev = np.zeros((emb_size, emb_size))
        M_prev = np.zeros((emb_size, 1))
        F_prev = np.zeros((emb_size, 1))
        t_prev = 0
        
        INITIAL_VALUES = {"A": A_prev, 'M': M_prev, "F": F_prev, "T": t_prev}
        self.all_embd_train = {}
        self.all_snaps_embs_train = {}
        
        # temporal aggregation
        for t in tqdm(self.train_t):
            # get window and find edge frequency
            data_window = get_frequency(self.train_data_group.get_group(t))
            self.all_embd_train,current_window_emb_train = self.model.inference(data_window, self.node_feat_map, t, 
                                                                                self.all_embd_train,INITIAL_VALUES, self.args.delta, self.args.alpha)

            # collect embeddings for each snapshot
            self.all_snaps_embs_train.update({t:current_window_emb_train})
    
    def _get_edge_embeddings(self,data_group,snaps, all_feats):
  
        embs = []

        for t in tqdm(snaps):
            
            feats = all_feats.get(t)
            hashmap_emb = {}

            df_win = data_group.get_group(t)
            for _, row in df_win.iterrows():
                edge_name = "_".join((str(int(row[0])),str(int(row[1])) ))
                emb = hashmap_emb.get(edge_name,None)
                
                if emb is None:
                    emb1 = np.concatenate((feats[str(int(row[0]))]['A'].flatten(),feats[str(int(row[0]))]['F'].flatten()))
                    emb2 = np.concatenate((feats[str(int(row[1]))]['A'].flatten(),feats[str(int(row[1]))]['F'].flatten()))
                    emb = np.concatenate((emb1,emb2))
                embs.append(emb)
            
        embs = np.vstack(embs)
        return embs
    
    def _train_vae(self):
        self.vae = VariationalAutoencoder(self.args,self.train_embs.shape[1],self.model_path)
        self.vae.train_vae(self.train_embs)
    
    def train(self):
        self._train_sirgn()
        self._get_train_node_embedding()
        self.train_embs = self._get_edge_embeddings(self.train_data_group,self.train_t,self.all_snaps_embs_train)
        self._train_vae()
        
    
    def test(self):
        pass
    
   
    
    def temporal_inference(self):
        pass
    

    
    def online_inference(self):
        pass
    
    
    
    
    
    